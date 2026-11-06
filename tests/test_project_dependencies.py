"""Dependency alignment and interface checks; does not install software."""
import json
from pathlib import Path

import yaml
from gsrefine.doctor import inspect_project

ROOT = Path(__file__).resolve().parents[1]


def test_single_env_declares_node22_and_required_converter():
    cfg = yaml.safe_load((ROOT / 'environment.yml').read_text())
    assert cfg['name'] == 'gsrefine'
    assert any(str(d).startswith('nodejs>=22,') for d in cfg['dependencies'])
    pkg = json.loads((ROOT / 'ui/package.json').read_text())
    assert 'playcanvas' in pkg['dependencies']
    assert '@playcanvas/splat-transform' in pkg['dependencies']
    assert '@playcanvas/splat-transform' not in pkg.get('optionalDependencies', {})


def test_doctor_reports_conversion_runtime_readonly():
    report = inspect_project()
    assert 'node_version' in report
    assert 'node_22_compatible' in report
    assert 'splat_transform_installed' in report
