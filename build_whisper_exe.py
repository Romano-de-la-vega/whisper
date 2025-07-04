"""
Utility script to build the portable Whisper application.
Run with `python build_whisper_exe.py` on a Windows machine to generate
a standalone .exe using PyInstaller.
"""

import subprocess
import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
APP_DIR = os.path.join(BASE_DIR, 'app transcription - github')
MAIN_SCRIPT = os.path.join(APP_DIR, 'opti whisper.py')  # main application
REQUIREMENTS = os.path.join(APP_DIR, 'requirements.txt')
ICON_FILE = os.path.join(APP_DIR, 'icon.ico')

def run(cmd):
    print(f'Running: {cmd}')
    subprocess.check_call(cmd, shell=True)


def main():
    # install dependencies
    run('python -m pip install --upgrade pip')
    run(f'pip install -r "{REQUIREMENTS}"')
    run('pip install pyinstaller')

    # build executable
    cmd = (
        f'pyinstaller --onefile --windowed --icon="{ICON_FILE}" '
        f'"{MAIN_SCRIPT}"'
    )
    run(cmd)
    print('\nExecutable created in the dist directory.')


if __name__ == '__main__':
    main()
