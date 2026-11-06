# Scene configuration policy

- `configs/songziguanyin.yaml` is a **tracked example** and baseline default.
- Create `configs/local/songziguanyin.yaml` for your machine's real absolute paths.
- `configs/local/` is ignored by Git and should never be overwritten by a code update.
- `--config` explicitly selects a scene configuration for all commands.
- Avoid hardcoding scene paths inside `gsrefine/*.py` and `ui/*.mjs`.
- Experiment outputs go under `runs/<scene>/...`, not into the source-code package.

One-time setup:

```bash
cd /home/xth/GSSceneRefine
mkdir -p configs/local
cp configs/songziguanyin.yaml configs/local/songziguanyin.yaml
# Edit configs/local/songziguanyin.yaml with correct existing source paths.
python run.py doctor --check-data --config configs/local/songziguanyin.yaml
```

For additional landscapes, add one matching tracked example and one separate local config; never create a new Conda environment or a separate project folder.
