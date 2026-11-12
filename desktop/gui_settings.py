"""Small desktop-only settings store. No .env access; Windows-user DPAPI key."""
import base64
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import urllib.parse


# Only these desktop settings may enter the isolated backend environment.
# Provider generation capacity includes reasoning; Agent limits bound visible text.
RUNTIME_INTS = {
    "LLM_PLANNER_MAX_GENERATION_TOKENS": (8192, 1, 65536),
    "LLM_ANSWER_MAX_GENERATION_TOKENS": (8192, 1, 65536),
    "LLM_TIMEOUT_SECONDS": (90, 1, 300),
    "AGENT_TOTAL_DEADLINE_MS": (300000, 1000, 300000),
    "AGENT_FINAL_ANSWER_RESERVE_MS": (90000, 100, 120000),
    "AGENT_MAX_PLANNER_TOKENS_PER_STEP": (1024, 64, 4096),
    "AGENT_MAX_TOTAL_PLANNER_TOKENS": (8192, 64, 32768),
    "AGENT_MAX_FINAL_ANSWER_TOKENS": (4096, 128, 16384),
}
RUNTIME_MODES = {
    "LLM_PLANNER_THINKING": {"enabled", "disabled", ""},
    "LLM_ANSWER_THINKING": {"enabled", "disabled", ""},
    "LLM_PLANNER_REASONING_EFFORT": {"low", "high", "max", ""},
    "LLM_ANSWER_REASONING_EFFORT": {"low", "high", "max", ""},
}


def desktop_runtime(config):
    """Migrate missing settings without overriding explicit values or reading .env."""
    supplied = config.get("agent_runtime", {})
    if not isinstance(supplied, dict):
        raise ValueError("agent_runtime must be an object")
    result = {}
    for key, (default, minimum, maximum) in RUNTIME_INTS.items():
        value = supplied.get(key, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Invalid desktop setting: {key}")
        result[key] = value
    # Known official Thinking API only. Other endpoints keep extension fields omitted
    # unless their compatibility has explicitly been configured by the user.
    llm = config.get("llm", {})
    host = urllib.parse.urlsplit(config.get("api_base", llm.get("base_url", ""))).hostname
    model = config.get("api_model", llm.get("model", ""))
    deepseek = host == "api.deepseek.com" and model in {"deepseek-flash", "deepseek-v4-pro"}
    for key, choices in RUNTIME_MODES.items():
        default = ("enabled" if key.endswith("THINKING") else "high") if deepseek else ""
        value = supplied.get(key, default)
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"Invalid desktop setting: {key}")
        result[key] = value
    for stage in ("PLANNER", "ANSWER"):
        if result[f"LLM_{stage}_THINKING"] == "disabled":
            # Preserve saved effort, but never send it alongside disabled thinking.
            result[f"LLM_{stage}_REASONING_EFFORT"] = ""
    if result["AGENT_FINAL_ANSWER_RESERVE_MS"] < result["LLM_TIMEOUT_SECONDS"] * 1000:
        raise ValueError("Final answer reserve must cover one API timeout")
    if result["AGENT_TOTAL_DEADLINE_MS"] <= result["AGENT_FINAL_ANSWER_RESERVE_MS"]:
        raise ValueError("Total deadline must exceed final answer reserve")
    if result["AGENT_MAX_TOTAL_PLANNER_TOKENS"] < result["AGENT_MAX_PLANNER_TOKENS_PER_STEP"]:
        raise ValueError("Total visible Planner budget must cover one step")
    return result


def protect(value: str, *, decrypt=False) -> str:
    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]
    raw = base64.b64decode(value) if decrypt else value.encode("utf-8")
    buffer = ctypes.create_string_buffer(raw)
    source = Blob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    api = ctypes.WinDLL("crypt32", use_last_error=True)
    function = api.CryptUnprotectData if decrypt else api.CryptProtectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise RuntimeError("无法读取此 Windows 用户保存的密钥，请在设置中重新输入。")
    try:
        result = ctypes.string_at(target.data, target.size)
        return result.decode("utf-8") if decrypt else base64.b64encode(result).decode("ascii")
    finally:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree(target.data)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def git_directory(value):
    path = Path(value).expanduser()
    if path.is_file() and path.name.lower() == "git.exe":
        return str(path.parent.resolve())
    for candidate in (path / "mingw64/bin", path, path / "cmd"):
        if (candidate / "git.exe").is_file():
            return str(candidate.resolve())
    return str(path.resolve()) if value else ""


class SettingsStore:
    def __init__(self, data: Path, defaults: dict):
        self.data = data
        self.path = data / "desktop-settings.json"
        self.defaults = {**defaults, "port": defaults.get("port", 18771)}

    def load(self):
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            result = {**self.defaults, **saved}
            if "agent_runtime" in self.defaults or "agent_runtime" in saved:
                base, user = self.defaults.get("agent_runtime", {}), saved.get("agent_runtime", {})
                if not isinstance(base, dict) or not isinstance(user, dict):
                    raise ValueError("agent_runtime must be an object")
                result["agent_runtime"] = {**base, **user}
            return result
        return dict(self.defaults)

    def public(self):
        value = self.load()
        return {key: value.get(key, "") for key in ("model_directory", "git_directory", "api_base", "api_model",
            "device", "batch_size", "max_length")} | {"data_directory": str(self.data),
            "key_saved": bool(value.get("api_key_dpapi"))}

    def save(self, incoming):
        if not isinstance(incoming, dict):
            raise ValueError("设置格式不正确。")
        current = self.load()
        for key in ("model_directory", "git_directory", "api_base", "api_model", "api_key"):
            if not isinstance(incoming.get(key, ""), str):
                raise ValueError("请填写文本格式的设置。")
        api_base = incoming.get("api_base", "").strip().rstrip("/")
        if api_base:
            try:
                url = urllib.parse.urlsplit(api_base)
                valid = url.scheme in {"http", "https"} and url.hostname and not (
                    url.username or url.password or url.query or url.fragment)
                url.port
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("API 地址需为 HTTP/HTTPS 地址，不能包含用户名、密钥、查询参数或片段。")
        current.update(model_directory=incoming.get("model_directory", "").strip(),
                       git_directory=git_directory(incoming.get("git_directory", "").strip()),
                       api_base=api_base, api_model=incoming.get("api_model", "").strip())
        if incoming.get("clear_key") is True:
            current.pop("api_key_dpapi", None)
        key = incoming.get("api_key", "").strip()
        if key:
            current["api_key_dpapi"] = protect(key)
        if "agent_runtime" in incoming:
            if not isinstance(incoming["agent_runtime"], dict):
                raise ValueError("agent_runtime must be an object")
            current["agent_runtime"] = {**current.get("agent_runtime", {}), **incoming["agent_runtime"]}
        desktop_runtime(current)  # Reject invalid budgets before persistence.
        atomic_json(self.path, current)
        return self.public()

    def runtime(self):
        config = self.load()
        key = protect(config["api_key_dpapi"], decrypt=True) if config.get("api_key_dpapi") else ""
        config["llm"] = {"base_url": config.get("api_base", ""), "model": config.get("api_model", ""), "api_key": key}
        return config

    def resource_errors(self):
        config = self.load()
        model = Path(config.get("model_directory", ""))
        errors = []
        required = ("config.json", "modules.json", "tokenizer_config.json")
        if not config.get("model_directory") or not all((model / item).is_file() for item in required):
            errors.append("BGE-M3 目录无效：请选择包含 config.json、modules.json 和分词器配置的完整权重目录。")
        elif not any(model.glob("*.bin")) and not any(model.glob("*.safetensors")):
            errors.append("BGE-M3 目录缺少模型权重文件，请选择完整的已下载目录。")
        if not (Path(config.get("git_directory", "")) / "git.exe").is_file():
            errors.append("未找到 Git：请选择 git.exe，或包含 Git 程序的安装目录。")
        return errors
