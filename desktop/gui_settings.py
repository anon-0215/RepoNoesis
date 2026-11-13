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


BAILIAN_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


def bailian_environment_key():
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as registry:
                key = winreg.QueryValueEx(registry, "DASHSCOPE_API_KEY")[0]
        except (ImportError, OSError):
            pass
    return key if isinstance(key, str) else ""


class SettingsValidationError(ValueError):
    """Only fixed, safe Chinese messages may cross the desktop bridge."""
    def __init__(self, message):
        self.message = message
        super().__init__(message)


def rerank_config(config):
    mode = config.get("rerank_mode")
    if mode is None:
        legacy = config.get("rerank_provider", "generation")
        mode = {"generation": "generation", "deepseek": "generation", "bailian": "independent"}.get(legacy) if isinstance(legacy, str) else None
    if mode not in ("generation", "independent"):
        raise SettingsValidationError("请选择“使用当前生成模型”或“使用独立重排服务”。")
    service = config.get("rerank_service", "bailian")
    if service != "bailian":
        raise SettingsValidationError("本轮独立重排仅支持阿里云百炼。")
    endpoint = config.get("bailian_endpoint", BAILIAN_ENDPOINT)
    model = config.get("bailian_model", "qwen3-rerank")
    timeout = config.get("bailian_timeout_seconds", 7)
    if not isinstance(endpoint, str):
        raise SettingsValidationError("百炼 API 地址需为文本。")
    try:
        p = urllib.parse.urlsplit(endpoint)
        host = p.hostname or ""
        valid = (p.scheme == "https" and p.port in {None,443} and not (
            p.username or p.password or p.query or p.fragment)
            and (host == "dashscope.aliyuncs.com" or (
                host.endswith(".cn-beijing.maas.aliyuncs.com")
                and host.split('.')[0].replace('-', '').isalnum()))
            and p.path == "/api/v1/services/rerank/text-rerank/text-rerank")
    except ValueError:
        valid = False
    if not valid:
        raise SettingsValidationError("百炼 API 地址须为受支持的北京官方 HTTPS 重排接口，不能包含密钥、用户名或查询参数。")
    if model != "qwen3-rerank":
        raise SettingsValidationError("本轮百炼重排仅支持模型 qwen3-rerank。")
    if type(timeout) not in {int,float} or not .1 <= timeout <= 15:
        raise SettingsValidationError("百炼超时需在0.1至15秒之间。")
    return {"rerank_mode": mode, "rerank_service": service,
            "rerank_provider": "bailian" if mode == "independent" else "generation",
            "bailian_endpoint": endpoint, "bailian_model": model, "bailian_timeout_seconds": timeout}


def rerank_key_source(value):
    if value.get("bailian_api_key_dpapi"):
        return "saved"
    if os.environ.get("DASHSCOPE_API_KEY"):
        return "process_environment"
    return "windows_user_environment" if bailian_environment_key() else "missing"


def enhancement_flags(config):
    # Invalid legacy values fail closed. In particular, "false" is not True.
    return {key: config.get(key, False) is True for key in ("hyde_enabled", "rerank_enabled")}


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
            # A user's explicit legacy selection wins over new release defaults.
            if "rerank_mode" not in saved and "rerank_provider" in saved:
                result.pop("rerank_mode", None)
            result.update(rerank_config(result))
            result.update(enhancement_flags(result))
            return result
        return {**self.defaults, **enhancement_flags(self.defaults), **rerank_config(self.defaults)}

    def public(self, value=None):
        value = self.load() if value is None else value
        source = rerank_key_source(value)
        missing = []
        if enhancement_flags(value)["rerank_enabled"]:
            if rerank_config(value)["rerank_mode"] == "independent":
                if source == "missing":
                    missing.append("百炼 API Key")
            else:
                missing = [label for field, label in (("api_base", "生成 API 地址"), ("api_model", "生成模型名称"),
                           ("api_key_dpapi", "生成 API Key")) if not value.get(field)]
        return {key: value.get(key, "") for key in ("model_directory", "git_directory", "api_base", "api_model",
            "device", "batch_size", "max_length")} | {"data_directory": str(self.data),
            "key_saved": bool(value.get("api_key_dpapi")), **enhancement_flags(value),
            **rerank_config(value), "bailian_key_saved": bool(value.get("bailian_api_key_dpapi")),
            "bailian_environment_key_present": bool(bailian_environment_key()),
            "bailian_key_source": source, "rerank_missing_fields": missing}

    def save(self, incoming):
        current = self.prepare(incoming)
        atomic_json(self.path, current)
        return self.public(current)

    def prepare(self, incoming):
        """Validate/merge without writing, so the host can first secure a safe restart."""
        if not isinstance(incoming, dict):
            raise ValueError("设置格式不正确。")
        current = self.load()
        for key in ("model_directory", "git_directory", "api_base", "api_model", "api_key"):
            if not isinstance(incoming.get(key, ""), str):
                raise ValueError("请填写文本格式的设置。")
        for key in ("hyde_enabled", "rerank_enabled"):
            if key in incoming:
                if type(incoming[key]) is not bool:
                    raise ValueError("检索增强开关必须为布尔值。")
                current[key] = incoming[key]
        api_base = incoming.get("api_base", current.get("api_base", "")).strip().rstrip("/")
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
        for field in ("model_directory", "git_directory", "api_model"):
            if field in incoming:
                current[field] = (git_directory(incoming[field].strip()) if field == "git_directory"
                                  else incoming[field].strip())
        if "api_base" in incoming:
            current["api_base"] = api_base
        if incoming.get("clear_key") is True:
            current.pop("api_key_dpapi", None)
        key = incoming.get("api_key", "").strip()
        if key:
            current["api_key_dpapi"] = protect(key)
        if "agent_runtime" in incoming:
            if not isinstance(incoming["agent_runtime"], dict):
                raise ValueError("agent_runtime must be an object")
            current["agent_runtime"] = {**current.get("agent_runtime", {}), **incoming["agent_runtime"]}
        for field in ("rerank_mode", "rerank_service", "rerank_provider", "bailian_endpoint", "bailian_model", "bailian_timeout_seconds"):
            if field in incoming:
                current[field] = incoming[field]
        if "rerank_provider" in incoming and "rerank_mode" not in incoming:
            current.pop("rerank_mode", None)
        current.update(rerank_config(current))
        if incoming.get("bailian_clear_key") is True:
            current.pop("bailian_api_key_dpapi", None)
        bailian_key = incoming.get("bailian_api_key", "")
        if not isinstance(bailian_key, str):
            raise ValueError("百炼密钥需为文本。")
        if bailian_key.strip():
            current["bailian_api_key_dpapi"] = protect(bailian_key.strip())
        desktop_runtime(current)  # Reject invalid budgets before persistence.
        return current

    def runtime(self):
        config = self.load()
        key = protect(config["api_key_dpapi"], decrypt=True) if config.get("api_key_dpapi") else ""
        config["llm"] = {"base_url": config.get("api_base", ""), "model": config.get("api_model", ""), "api_key": key}
        config["bailian_api_key"] = (protect(config["bailian_api_key_dpapi"], decrypt=True)
                                     if config.get("bailian_api_key_dpapi") else bailian_environment_key())
        config.update(rerank_config(config))
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
