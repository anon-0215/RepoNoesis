"""Assemble a clean portable ZIP from an explicit program-only scope.

Never visits sample user-data, .env, databases or source repositories. Reuses
runtime files without installing/upgrading dependencies. Refuses an existing
ZIP; --resume can finish an interrupted program-only assembly.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import zipfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_files import DESKTOP_FILES

HERE = Path(__file__).resolve().parent
CACHE_DIRS = {"__pycache__", ".cache", ".pytest_cache", ".mypy_cache"}
FORBIDDEN_NAMES = {".env", "desktop-settings.json", "acceptance-ledger.json", "id_rsa", "id_ed25519"}
FORBIDDEN_DIRS = {"user-data", ".git", ".venv", "EBWebView", "logs", "screenshots"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_exclusion(relative):
    if any(part in CACHE_DIRS for part in relative.parts) or relative.suffix.lower() in {".pyc", ".pyo"}:
        return "generated Python/tool cache"
    if relative == Path("conda-meta/history") or relative.name == "direct_url.json":
        return "build provenance/temp record"
    if relative.parts and relative.parts[0].startswith("tmp"):
        return "runtime build temporary file"
    return None


def check_relative(relative):
    if any(part in FORBIDDEN_DIRS for part in relative.parts):
        raise RuntimeError(f"Forbidden directory in distribution: {relative}")
    if relative.name in FORBIDDEN_NAMES or relative.name.startswith(".env."):
        raise RuntimeError(f"Forbidden private file in distribution: {relative}")
    if relative.suffix.lower() in {".sqlite", ".sqlite3", ".db", ".log", ".lnk"}:
        raise RuntimeError(f"Unexpected data/log/link in distribution: {relative}")


def files_under(root):
    for directory, dirs, files in os.walk(root, followlinks=False):
        directory = Path(directory)
        for name in dirs + files:
            path = directory / name
            if path.is_symlink() or path.lstat().st_file_attributes & 0x400:
                raise RuntimeError(f"Reparse point must be materialized explicitly: {path.relative_to(root)}")
        for name in files:
            yield directory / name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-app", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Finish a program-only stage when no ZIP exists")
    parser.add_argument("--version-suffix", default="", help="Optional safe archive version suffix, e.g. -v2")
    parser.add_argument("--frontend-dist", type=Path, help="Explicit desktop production build directory")
    args = parser.parse_args()
    sample, output = args.sample_app.resolve(), args.output.resolve()
    if not (sample / "runtime/python.exe").is_file():
        raise RuntimeError("Sample app/runtime/python.exe not found")
    if not output.is_relative_to(sample.parents[1]) or output == sample or output.is_relative_to(sample):
        raise RuntimeError("Output must be a separate directory under the workspace output directory")
    stage = output / "package/源鉴便携版"
    if args.version_suffix and not re.fullmatch(r"[-A-Za-z0-9_]{1,24}", args.version_suffix):
        raise RuntimeError("Invalid archive version suffix")
    archive_path = output / f"源鉴-Windows-便携试用{args.version_suffix}.zip"
    if archive_path.exists() or (stage.exists() and not args.resume):
        raise RuntimeError("ZIP exists or stage needs explicit --resume; use a fresh output directory")
    stage.mkdir(parents=True, exist_ok=args.resume)
    app = stage / "app"
    excluded = []
    copied = 0

    def copy(original, relative):
        nonlocal copied
        check_relative(relative)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != original.stat().st_size or sha256(target) != sha256(original):
            shutil.copy2(original, target)
        copied += 1
        if copied % 10000 == 0:
            print(f"Copied {copied} program/runtime files", flush=True)

    # The only source scopes: runtime, backend/app Python, built frontend,
    # six named desktop modules, static launchers/docs and safe metadata.
    for original in files_under(sample / "runtime"):
        relative = original.relative_to(sample / "runtime")
        reason = runtime_exclusion(relative)
        if reason:
            excluded.append({"path": f"app/runtime/{relative.as_posix()}", "reason": reason})
        else:
            copy(original, Path("app/runtime") / relative)
    for original in sorted((HERE.parent / "backend/app").rglob("*.py")):
        copy(original, Path("app/backend/app") / original.relative_to(HERE.parent / "backend/app"))
    frontend = (args.frontend_dist or HERE.parent / "frontend/dist").resolve()
    if not (frontend / "index.html").is_file():
        raise RuntimeError("Build and validate the current desktop frontend before packaging")
    for original in files_under(frontend):
        copy(original, Path("app/frontend") / original.relative_to(frontend))
    for name in DESKTOP_FILES:
        copy(HERE / name, Path("app/desktop") / name)
    copy(HERE / "desktop-config.portable.json", Path("app/desktop-config.json"))
    copy(sample / "DEPENDENCY_VERSIONS.json", Path("app/DEPENDENCY_VERSIONS.json"))
    copy(HERE / "PortableStart.cmd", Path("排错启动.cmd"))
    # WSH uses UTF-16 BOM for reliable Chinese text, including error dialogs.
    (stage / "源鉴.vbs").write_text((HERE / "PortableLaunch.vbs").read_text(encoding="utf-8"), encoding="utf-16")
    for name in ("操作卡.md", "依赖说明.md"):
        copy(HERE / "portable-docs" / name, Path(name))
        shutil.copy2(stage / name, output / name)
    card = (stage / "操作卡.md").read_text(encoding="utf-8")
    plain = card.replace("# ", "").replace("**", "").replace("`", "")
    (stage / "操作卡.txt").write_text(plain, encoding="utf-8-sig")
    shutil.copy2(stage / "操作卡.txt", output / "操作卡.txt")
    baseline = subprocess.check_output(["git", "-c", f"safe.directory={HERE.parent.as_posix()}",
        "-C", str(HERE.parent), "rev-parse", "HEAD"], text=True).strip()
    (app / "PORTABLE_BUILD.json").write_text(json.dumps({
        "baseline": baseline, "kind": "Windows x64 portable trial", "window": "pywebview",
        "runtime": "reused selected sample runtime, no dependency upgrades",
        "data_location": "../user-data (relative to app)", "defaults": "empty local resources and API",
        "native_layout_repair_retained": True,
    }, indent=2), encoding="utf-8")
    manifest = []
    licenses = []
    for file in sorted(files_under(stage)):
        relative = file.relative_to(stage)
        if relative == Path("文件清单.tsv"):
            continue  # --resume must not hash a previous manifest into itself.
        check_relative(relative)
        manifest.append({"path": relative.as_posix(), "bytes": file.stat().st_size, "sha256": sha256(file)})
        if any(token in file.name.lower() for token in ("license", "copying", "notice", "copyright")):
            licenses.append(relative.as_posix())
    assert licenses, "Missing upstream licenses"
    listing = "sha256\tbytes\tpath\n" + "".join(f"{r['sha256']}\t{r['bytes']}\t{r['path']}\n" for r in manifest)
    (stage / "文件清单.tsv").write_text(listing, encoding="utf-8-sig")
    shutil.copy2(stage / "文件清单.tsv", output / "文件清单.tsv")
    print(f"Compressing {len(manifest) + 1} files; no personal data scope was visited", flush=True)
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for file in sorted(files_under(stage)):
            archive.write(file, (Path(stage.name) / file.relative_to(stage)).as_posix())
    checksum = sha256(archive_path)
    (output / "SHA256SUMS.txt").write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")
    info = {"zip": archive_path.name, "sha256": checksum, "zip_bytes": archive_path.stat().st_size,
        "uncompressed_bytes": sum(row["bytes"] for row in manifest) + (stage / "文件清单.tsv").stat().st_size,
        "file_count": len(manifest) + 1, "license_file_count": len(licenses),
        "manifest_self_excluded": True, "symlinks_or_reparse_points": 0}
    (output / "PACKAGE.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (output / "records").mkdir(exist_ok=True)
    (output / "records/build-scope.json").write_text(json.dumps({
        "runtime_source": str(sample / "runtime"), "backend_source": str(HERE.parent / "backend/app"),
        "frontend_source": str(frontend), "desktop_source": str(HERE),
        "user_data_visited": False, "excluded": excluded, "preserved_licenses": licenses,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
