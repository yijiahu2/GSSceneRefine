"""Launch a Windows-native model stream while preserving one WSL gsrefine env."""
from __future__ import annotations
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def to_windows_path(path):
    """Use WSL's own mount mapping; never interpret or guess Windows drive letters."""
    p=Path(path).resolve(strict=True)
    if not shutil.which('wslpath'):
        raise RuntimeError('Native Windows file service is only available from WSL; wslpath missing')
    return subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()


def native_service_command(settings):
    if not shutil.which('powershell.exe'):
        raise RuntimeError('WSL Windows PowerShell interop unavailable; verify powershell.exe and WSL interop')
    script=ROOT/'windows'/'serve_model.ps1'
    return ['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
            '-File',to_windows_path(script),'-ModelPath',to_windows_path(settings.gaussian_ply),
            '-Port',str(settings.file_port),'-AllowedOrigin',f'http://127.0.0.1:{settings.ui_port}']


def start_native_server(settings):
    command=native_service_command(settings)
    logs=settings.out_dir
    logs.mkdir(parents=True,exist_ok=True)
    logfile=logs/'windows_model_service.log'
    stream=logfile.open('a',encoding='utf8')
    # Running from a Windows-mounted cwd avoids Windows launching cmd in a UNC working directory.
    windows_cwd=Path('/mnt/c/Windows')
    if not windows_cwd.is_dir():windows_cwd=settings.gaussian_ply.parent
    try:
        proc=subprocess.Popen(command,cwd=windows_cwd,stdout=stream,stderr=subprocess.STDOUT)
    finally:
        stream.close()
    time.sleep(.5)
    if proc.poll() is not None:
        raise RuntimeError(f'Windows file service exited immediately (status {proc.returncode}). Inspect {logfile}')
    return proc,logfile
