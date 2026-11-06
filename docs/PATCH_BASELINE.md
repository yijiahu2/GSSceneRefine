# Patch compatibility

This update was prepared against the public repository:

- `https://github.com/yijiahu2/GSSceneRefine`
- branch: `main`
- Git tree SHA: `d8efce54808f3c57d4b5ef85ad7acde1fb0465a7`
- Date: 2026-11-06

The patch updates only the orchestration/output contract and the corresponding test/documentation files. It keeps the existing Stage-A ROI-anchored geodesic recovery and Stage-B adaptive Gaussian safety logic as implemented in the above Git tree. It does not create a Git commit or push changes to the remote repository.

Before installing over local work, back up any manually modified source code. `unzip -o` replaces only the named files; other project files and historical run data are untouched.
