"""Optional, zero-recompute Stage-1 -> Stage-2 trusted ROI surface handoff.

Stage 1 remains authoritative. This module ONLY preserves a face mask that Stage 1
has already produced, with explicit Poisson/ROI fingerprints and final-R3
semantics. It NEVER estimates a missing mask or modifies Stage-1 algorithms.

Stage-1 study writers can create the following in their temporary work folder:
  stage2_ready_roi_surface_faces.npz
fields:
  keep: bool vector indexed by ORIGINAL trusted Poisson triangle rows
  poisson_sha256, roi_sha256: scalar UTF-8 strings
  selection_semantics: exact value 'final_r3_trusted_roi_surface_faces'

Stage-1 publication calls retain_stage1_surface_faces before work files are
cleaned. If no validated record exists, the next stage may use its separately
labeled, bounded geometric fallback; absence is not an error.
"""
from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np

from gsrefine.roi.core import file_sha256

SOURCE_NAME = 'stage2_ready_roi_surface_faces.npz'
PUBLISHED_NAME = 'trusted_roi_surface_faces.npz'
SEMANTICS = 'final_r3_trusted_roi_surface_faces'


def read_verified_surface_faces(path, poisson, roi, triangle_count=None):
    """Validate an *exact* Stage-1 generated original-mesh face selection.

    A Gaussian keep mask, initial polygons, or a cropped secondary mesh is NOT
    interchangeable with this full-length Poisson triangle-row mask.
    """
    path, poisson, roi = map(Path, (path, poisson, roi))
    if not all(p.is_file() for p in (path, poisson, roi)):
        raise FileNotFoundError('Stage-1 ROI surface handoff/source file is missing')
    with np.load(path, allow_pickle=False) as record:
        mandatory = ('keep', 'poisson_sha256', 'roi_sha256', 'selection_semantics')
        if any(k not in record for k in mandatory):
            raise ValueError('Stage-1 ROI surface handoff lacks certified semantics/hashes')
        mask = np.asarray(record['keep'])
        ps = str(record['poisson_sha256'].item())
        rs = str(record['roi_sha256'].item())
        semantics = str(record['selection_semantics'].item())
    if semantics != SEMANTICS:
        raise ValueError('Stage-1 ROI face record is NOT a final R3 trusted-surface selection')
    if mask.dtype != np.dtype('bool') or mask.ndim != 1 or not mask.any():
        raise ValueError('Stage-1 ROI surface mask must be a nonempty bool vector')
    if triangle_count is not None and mask.shape != (int(triangle_count),):
        raise ValueError('Stage-1 face mask must index original Poisson triangle rows')
    if ps != file_sha256(poisson) or rs != file_sha256(roi):
        raise ValueError('Stage-1 ROI surface source files changed; mask cannot be reused')
    return mask


def save_computed_stage1_surface_faces(work, keep, poisson, roi, *, face_count=None):
    """Called by Stage 1 *only if* it already computed exact final-R3 faces.

    Does not calculate or infer geometry. It preserves the current original
    Poisson triangle order so Stage 2 can use the mask without recalculating it.
    """
    work, poisson, roi = map(Path, (work, poisson, roi))
    mask = np.asarray(keep)
    if mask.ndim != 1 or mask.dtype != np.dtype('bool') or not mask.any():
        raise ValueError('Stage-1 final trusted ROI faces must be a nonempty bool array')
    if face_count is not None and len(mask) != int(face_count):
        raise ValueError('Stage-1 surface mask length must match original Poisson faces')
    dest = work / SOURCE_NAME
    if dest.exists():
        raise FileExistsError('Stage-1 face handoff already generated; refusing overwrite')
    work.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.stage2_face_work_', suffix='.npz', dir=work)
    os.close(fd)
    try:
        np.savez_compressed(name, keep=mask,
                            poisson_sha256=np.array(file_sha256(poisson)),
                            roi_sha256=np.array(file_sha256(roi)),
                            selection_semantics=np.array(SEMANTICS))
        os.replace(name, dest)
    finally:
        Path(name).unlink(missing_ok=True)
    return dest


def retain_stage1_surface_faces(work, metadata_dir, poisson, roi):
    """Publish only an already-generated and certified Stage-1 mask.

    This function is intentionally a NO-OP when Stage-1 study has not exposed
    such a face mask. No expensive geometry work is performed to fill gaps.
    """
    work, metadata_dir, poisson, roi = map(Path, (work, metadata_dir, poisson, roi))
    source = work / SOURCE_NAME
    if not source.exists():
        return None
    import trimesh
    raw = trimesh.load(poisson, process=False, force='mesh')
    if not isinstance(raw, trimesh.Trimesh):
        raise ValueError('Stage-1 Poisson is not a triangle mesh')
    keep = read_verified_surface_faces(source, poisson, roi, len(raw.faces))
    # Compact, explicit, canonical publication. No overwrite on retry.
    metadata_dir.mkdir(parents=True, exist_ok=True)
    dest = metadata_dir / PUBLISHED_NAME
    if dest.exists():
        raise FileExistsError(f'Existing Stage-1 ROI surface record: {dest}')
    fd, temporary = tempfile.mkstemp(prefix='.stage2_surface_', suffix='.npz', dir=metadata_dir)
    os.close(fd)
    try:
        np.savez_compressed(temporary, keep=keep,
                            poisson_sha256=np.array(file_sha256(poisson)),
                            roi_sha256=np.array(file_sha256(roi)),
                            selection_semantics=np.array(SEMANTICS))
        os.replace(temporary, dest)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return dest.resolve()
