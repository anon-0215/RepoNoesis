"""Build same-origin desktop UI with child-only flags, including Windows PowerShell 5.1."""
import argparse
import os
from pathlib import Path
import subprocess


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--node', default='node', help='Build-time Node executable')
    args = parser.parse_args(argv)
    frontend = Path(__file__).resolve().parents[1] / 'frontend'
    environment = dict(os.environ, VITE_API_BASE_URL='', VITE_DESKTOP='true')
    subprocess.run([args.node, 'node_modules/typescript/bin/tsc'], cwd=frontend, env=environment, check=True)
    subprocess.run([args.node, 'node_modules/vite/bin/vite.js', 'build', '--configLoader', 'native',
                    '--outDir', str(args.output.resolve())], cwd=frontend, env=environment, check=True)


if __name__ == '__main__':
    main()
