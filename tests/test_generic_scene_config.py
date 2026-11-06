"""Generic configuration can be reused without scene-specific file paths."""
from pathlib import Path
import yaml
from gsrefine.geometry.prepare import prepare_input_config


def test_generic_config_is_path_free_and_retargets_scenes(tmp_path):
    default = Path(__file__).resolve().parents[1] / 'configs' / 'default.yaml'
    before = default.read_bytes()
    data = yaml.safe_load(before)
    assert data['paths'] == {}
    assert data['scene'] == {'name': 'template', 'method': 'template'}
    assert data['roi_init']['orientation']['mode'] == 'auto'

    root = tmp_path / 'results'
    for scene, method in [('bazhaifeng', 'absgs'), ('sengmaofeng', 'vanilla')]:
        inp = tmp_path / scene / 'sparse'
        inp.mkdir(parents=True)
        points, images = inp / 'points3D.bin', inp / 'images.bin'
        points.write_bytes(b'fixture')
        images.write_bytes(b'fixture')
        model = tmp_path / scene / method / 'point_cloud.ply'
        model.parent.mkdir(parents=True)
        model.write_bytes(b'fixture')
        effective = prepare_input_config(
            default, scene=scene, method=method,
            colmap_points=str(points), colmap_images=str(images),
            gaussian_ply=str(model), final_output_root=root)
        runtime = yaml.safe_load(effective.read_text())
        assert effective.is_relative_to(root / scene / 'geometry')
        assert runtime['scene'] == {'name': scene, 'method': method}
        assert runtime['module1']['mode'] == 'auto'
        assert runtime['roi_init']['orientation']['frame_id'] == f'{scene}_colmap_original'
        assert runtime['roi_init']['orientation']['colmap_images'] == str(images)
        assert runtime['workflow']['output_root'] == str(root)
        assert Path(runtime['roi_init']['output_dir']).is_relative_to(root)
        assert Path(runtime['module1']['output_dir']).is_relative_to(root)
    assert default.read_bytes() == before
