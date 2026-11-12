"""Regression for restored validated V2 flood + parity, independent of Open3D.

The production pipeline is still responsible for R3 provenance and complete
sealless/SfM geometry checks.  These tests protect the *interface* correction,
not claim that a particular 3D scene is topologically certified.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from gsrefine.cleanup import roi_barrier_parity as parity
from gsrefine.cleanup import stage2_onepass as onepass


def _audit_setup(tmp_path, *, coarse="NO_TOP_TO_BOTTOM_PATH_SEEN_NOT_PROVEN_SEALED",
                 fine="NO_TOP_TO_BOTTOM_PATH_SEEN_NOT_PROVEN_SEALED", touching=0):
    audit = tmp_path / "audit"
    audit.mkdir()
    manifest = tmp_path / "manifest.json"
    r3 = tmp_path / "r3.npz"
    poisson = tmp_path / "poisson.ply"
    gaussian = tmp_path / "gaussian_roi.ply"
    for p in (manifest, r3, poisson, gaussian):
        p.write_bytes(p.name.encode())
    data = {
        "schema": "GSSceneRefine.final_R3_global_Poisson_barrier_audit.v1",
        "status": "FLOOD_SAMPLING_COMPLETED_NOT_CERTIFIED_SEALED",
        "inputs": {
            "manifest_sha256": parity._sha(manifest),
            "r3_sha256": parity._sha(r3),
            "original_poisson_sha256": parity._sha(poisson),
            "final_roi_gaussian": str(gaussian),
        },
        "r3_verified_replay": {"v3_handoff_max_field_distance_error": 0.0},
        "flood": {
            "coarse": {"status": coarse, "grid": {"roi_touching_audit_window_samples": touching,
                                                 "voxel": 0.3}},
            "fine": {"status": fine, "grid": {"roi_touching_audit_window_samples": touching,
                                               "voxel": 0.17}},
        },
    }
    (audit / "roi_global_poisson_barrier_report.json").write_text(json.dumps(data), encoding="utf8")
    (audit / "global_poisson_final_r3_intersection.npz").write_bytes(b"dummy")
    for name in ("coarse_connectivity.npz", "fine_connectivity.npz"):
        (audit / name).write_bytes(b"dummy")
    inputs = {"gaussian_ply": gaussian, "poisson_mesh": poisson}
    return audit, manifest, r3, poisson, gaussian, inputs


def test_exact_successful_gate_passes_for_both_levels(tmp_path, monkeypatch):
    audit, manifest, r3, _, _, inputs = _audit_setup(tmp_path)
    expected = {"closed_cycles": 1, "intersections": 1857}
    monkeypatch.setattr(parity, "_validate_intersection_loop", lambda _: expected)
    result, loops = parity._audit_identity(audit, manifest, inputs, r3)
    assert loops == expected
    assert result["flood"]["coarse"]["status"].startswith("NO_TOP")
    assert result["flood"]["fine"]["status"].startswith("NO_TOP")
    assert not (audit / "region_separation_gate.json").exists()


@pytest.mark.parametrize("which", ["coarse", "fine"])
def test_old_failed_flood_still_blocks(tmp_path, monkeypatch, which):
    kwargs = {which: "TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH"}
    audit, manifest, r3, _, _, inputs = _audit_setup(tmp_path, **kwargs)
    monkeypatch.setattr(parity, "_validate_intersection_loop", lambda _: {"closed_cycles": 1})
    with pytest.raises(ValueError, match=f"Barrier flood not suitable for parity classification: {which}"):
        parity._audit_identity(audit, manifest, inputs, r3)


def test_source_mismatch_and_window_touch_still_block(tmp_path, monkeypatch):
    audit, manifest, r3, poisson, gaussian, inputs = _audit_setup(tmp_path)
    monkeypatch.setattr(parity, "_validate_intersection_loop", lambda _: {"closed_cycles": 1})
    poisson.write_bytes(b"modified!")
    with pytest.raises(ValueError, match="source mismatch"):
        parity._audit_identity(audit, manifest, inputs, r3)
    poisson.write_bytes(poisson.name.encode())
    data = json.loads((audit/"roi_global_poisson_barrier_report.json").read_text())
    data["flood"]["coarse"]["grid"]["roi_touching_audit_window_samples"] = 1
    (audit/"roi_global_poisson_barrier_report.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="touches truncated"):
        parity._audit_identity(audit, manifest, inputs, r3)


def test_verified_complete_audit_reused_old_failure_not_reused(tmp_path):
    audit, manifest, r3, poisson, _, _ = _audit_setup(tmp_path)
    # The one-pass finds candidates recursively under ROI_run/cleanup.
    roi_run = tmp_path / "roi_run"
    target = roi_run / "cleanup" / "diagnostics" / "audit"
    target.mkdir(parents=True)
    for p in audit.iterdir():
        (target/p.name).write_bytes(p.read_bytes())
    assert onepass._find_barrier_audit(roi_run, manifest, r3, poisson) == target
    record = json.loads((target/"roi_global_poisson_barrier_report.json").read_text())
    record["flood"]["fine"]["status"] = "TOP_TO_BOTTOM_PATH_FOUND_SAMPLED_GRAPH"
    (target/"roi_global_poisson_barrier_report.json").write_text(json.dumps(record))
    assert onepass._find_barrier_audit(roi_run, manifest, r3, poisson) is None


def test_sealless_path_and_v2_cache_contract():
    src = Path(parity.__file__).read_text(encoding="utf8")
    total = Path(onepass.__file__).read_text(encoding="utf8")
    assert "from gsrefine.cleanup.stage2_geometry_validation import validate_parity_geometry" in src
    assert "from gsrefine.cleanup.open_ab_main" not in src
    assert "barrier_separation_gate" not in src
    assert "region_separation_gate" not in src
    assert parity.VERSION == "GSSceneRefine.roi_barrier_parity_sdf.v2"
    assert total.count("'GSSceneRefine.roi_barrier_parity_sdf.v2'") == 2
    assert "'GSSceneRefine.roi_barrier_parity_sdf.v3'" not in total
    assert "single-ring" in total


def test_other_gaussian_rules_identical_to_v2_reference_if_available():
    """Optional exact regression against original V2 classifier source bundle."""
    import os
    reference = os.environ.get("GSREFINE_V2_PARITY_REFERENCE")
    if not reference:
        pytest.skip("V2 reference bundle not installed in user's source tree")
    ref = Path(reference).read_text(encoding="utf8")
    now = Path(parity.__file__).read_text(encoding="utf8")
    def functions(text):
        tree = ast.parse(text)
        return {node.name: ast.get_source_segment(text, node)
                for node in tree.body if isinstance(node, ast.FunctionDef)}
    ref_functions = functions(ref)
    now_functions = functions(now)
    for name in ("_validate_intersection_loop", "_audit_identity", "_make_navigation",
                 "_query_nearby_candidates", "segment_parity", "_crossings_from_hits"):
        assert now_functions[name] == ref_functions[name], name
