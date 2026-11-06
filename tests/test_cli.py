"""Standalone project entrypoint regression tests."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def test_umbrella_cli_names():
    result=subprocess.run([sys.executable,str(ROOT/'run.py'),'--help'],capture_output=True,text=True,cwd=ROOT)
    assert result.returncode == 0
    assert 'GSSceneRefine' in result.stdout
    assert 'roi' in result.stdout and 'poisson' in result.stdout

def test_roi_cli_help():
    result=subprocess.run([sys.executable,str(ROOT/'run.py'),'roi','--help'],capture_output=True,text=True,cwd=ROOT)
    assert result.returncode == 0
    assert '--config' in result.stdout
