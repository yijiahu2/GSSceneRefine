"""The project diagnostic must remain read-only and work without scene data."""
from pathlib import Path
import subprocess
import sys

from gsrefine.doctor import ROOT, inspect_project


def test_doctor_project_structure():
    report = inspect_project()
    assert report['project_root'] == str(ROOT)
    assert report['expected_conda_env'] == 'gsrefine'
    assert all(report['source_tree'].values())
    assert 'data_inputs' not in report


def test_doctor_json_no_data():
    p = subprocess.run(
        [sys.executable, str(ROOT / 'run.py'), 'doctor', '--no-config', '--json'],
        text=True, capture_output=True, cwd=ROOT,
    )
    assert p.returncode == 0, p.stderr
    import json
    data = json.loads(p.stdout)
    assert data['expected_conda_env'] == 'gsrefine'
    assert 'data_inputs' not in data
