"""Build using the selected, preinstalled product Conda environment.

Install conda-pack into an isolated --target directory, then pass --tools.
Node/npm and frontend/node_modules are build dependencies only.
"""
import argparse
import functools
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_files import copy_application


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-runtime", action="store_true")
    parser.add_argument("--frontend-dist", type=Path, help="Explicit desktop production build directory")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == root or root.is_relative_to(output):
        raise RuntimeError("Output must be separate from the source root")
    if output.exists() and any(output.iterdir()) and not args.reuse_runtime:
        raise RuntimeError("Use a fresh output directory; --reuse-runtime is only for an explicit development copy")
    output.mkdir(parents=True, exist_ok=True)
    if not args.reuse_runtime:
        sys.path.insert(0, str(args.tools.resolve()))
        import conda_pack
        from conda_pack.formats import NoArchive
        # no-archive otherwise hardlinks on the same drive; copies keep the
        # product environment untouched during relocation and later updates.
        original = NoArchive.__init__
        def initialize(self, *a, **kw):
            original(self, *a, **kw)
            self.copy_func = functools.partial(shutil.copy2, follow_symlinks=False)
        NoArchive.__init__ = initialize
        # conda-pack 0.9.2 returns a truthy object from this context manager,
        # suppressing copy failures. Fail the build instead.
        NoArchive.__exit__ = lambda self, *a: False
        env = conda_pack.CondaEnv.from_prefix(sys.prefix)
        env = env.exclude("**/__pycache__/*").exclude("*.pdb").exclude("node.exe")
        env.pack(output=str(output / "runtime"), format="no-archive", verbose=False)
        scripts = output / "runtime/Scripts"
        subprocess.run([str(output / "runtime/python.exe"), "-I", "-B", "-c",
            "import sys,runpy; sys.path.insert(0,sys.argv[1]); runpy.run_module('conda-unpack-script',run_name='__main__')",
            str(scripts)], check=True)
    required = ["python.exe", "pythonw.exe", "python312.dll", "Scripts/conda-unpack-script.py",
                "Lib/site-packages/torch/__init__.py", "Lib/site-packages/webview/__init__.py",
                "Lib/site-packages/pythonnet/__init__.py", "Lib/site-packages/clr_loader/__init__.py"]
    for name in required:
        if not (output / "runtime" / name).is_file():
            raise RuntimeError(f"Runtime incomplete: {name}")
    copy_application(root, output, frontend_dist=args.frontend_dist)
    inventory = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions(
        path=[str(output / "runtime/Lib/site-packages")])}
    (output / "DEPENDENCY_VERSIONS.json").write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
    baseline = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
                                        "rev-parse", "HEAD"], cwd=root, text=True).strip()
    (output / "BUILD.json").write_text(json.dumps({"baseline": baseline, "python": sys.version,
        "conda_pack": "reused" if args.reuse_runtime else conda_pack.__version__, "route": "copied relocatable Conda Python; not frozen",
        "runtime_files": sum(1 for p in (output / "runtime").rglob("*") if p.is_file())}, indent=2), encoding="utf-8")
    print("Build complete:", output)


if __name__ == "__main__":
    main()
