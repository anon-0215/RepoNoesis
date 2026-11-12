"""One explicit application copy scope shared by initial and refresh builds."""
from pathlib import Path
import shutil

DESKTOP_FILES = ('desktop_host.py', 'gui_settings.py', 'launcher.py', 'server.py', 'shell.html', 'acceptance.py', 'settings_lifecycle.py')


def copy_application(source: Path, target: Path, *, frontend_dist: Path | None = None):
    source, target = source.resolve(), target.resolve()
    frontend = (frontend_dist or source / 'frontend/dist').resolve()
    if target == source or source.is_relative_to(target):
        raise RuntimeError('Application output must be separate from the source root')
    required = [source / 'backend/app/main.py', frontend / 'index.html',
                *(source / 'desktop' / name for name in DESKTOP_FILES),
                *(source / 'desktop' / name for name in ('Start.cmd', 'Launch.vbs', 'desktop-config.example.json'))]
    if not all(path.is_file() for path in required):
        raise RuntimeError('Application source or production frontend is incomplete; build the desktop frontend first')
    if frontend == target or frontend.is_relative_to(target) or target.is_relative_to(frontend):
        raise RuntimeError('Frontend input must be separate from the application output')
    target.mkdir(parents=True, exist_ok=True)
    for original, destination in ((source / 'backend/app', target / 'backend/app'), (frontend, target / 'frontend')):
        shutil.copytree(original, destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (target / 'desktop').mkdir(exist_ok=True)
    for name in DESKTOP_FILES:
        shutil.copy2(source / 'desktop' / name, target / 'desktop' / name)
    shutil.copy2(source / 'desktop/Start.cmd', target / 'Start.cmd')
    # WSH needs a BOM for its Chinese filename/text on Windows.
    (target / '源鉴.vbs').write_text((source / 'desktop/Launch.vbs').read_text(encoding='utf-8'), encoding='utf-16')
    if not (target / 'desktop-config.json').exists():
        shutil.copy2(source / 'desktop/desktop-config.example.json', target / 'desktop-config.json')
