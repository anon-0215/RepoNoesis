"""Explicitly replace application files in a stopped development copy."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app_files import copy_application


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frontend-dist', type=Path)
    parser.add_argument('--replace-app', required=True, action='store_true',
                        help='Acknowledge replacing application files in this stopped development copy')
    args = parser.parse_args(argv)
    target = args.output.resolve()
    if not (target / 'runtime/python.exe').is_file():
        raise RuntimeError('Prepare a separate runtime copy with window-requirements.txt first')
    copy_application(Path(__file__).resolve().parents[1], target, frontend_dist=args.frontend_dist)
    (target / 'DEPENDENCY_VERSIONS.json').write_text(json.dumps({d.metadata['Name']: d.version for d in
        importlib.metadata.distributions(path=[str(target / 'runtime/Lib/site-packages')])}, indent=2), encoding='utf-8')
    print('Application files refreshed; runtime and existing config preserved')


if __name__ == '__main__':
    main()
