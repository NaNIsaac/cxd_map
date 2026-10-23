# cxd_map

This repository keeps the implementation as it was before it was imported into [CXD](https://github.com/NaNIsaac/CXD) with its history on 2026-10-16 (CXD f669861, moved to `cxd.mapping` in eeb127f). There, `mapping.py` is split into `cxd.mapping.spherical` and `qorageom` (`shear_correction`, `rotation_matrix`), `analysis.py` is `cxd.mapping.analysis`, `visualize.py` is `cxd.mapping.vtk_view`, the line-profile code of `main.py` is `cxd.mapping.profiles` and `scripts/streak_profiles.py`, `load_roi_mat` is `qorageom.io.load_mat` and `center_of_mass` is `cxd.core.statistics.get_com`. `map2solid_angle`, used only in `notebooks/test.ipynb`, was not carried over.

Bragg coherent X-ray diffraction (BCDI) peak analysis for one 3D diffraction pattern. It maps the pattern around its center of mass onto a (cos θ, φ) angular histogram, finds streak and facet directions in that map with DBSCAN, plots intensity line profiles along pairs of opposite directions, and renders the pattern in reciprocal-space coordinates with VTK.

A Korean version of this file is [README.ko.md](README.ko.md).

## Requirements

Python 3.13 or later and numpy, scipy, scikit-learn, matplotlib and vtk. `pyproject.toml` also lists napari, pyvista and trame, which the code does not import.

## Usage

```powershell
uv sync
$env:PYTHONPATH = "src"     # pyproject.toml has no build-system, so cxd_map is not installed
uv run main.py
```

Data paths are relative to the `DATA` environment variable. Edit `data_dir`, `save_dir` and `run_ns` in `main.py` (the default input is `$DATA\260324_9C\onsite\ROI\scan_0255.mat`). The input is a MATLAB `.mat` file whose `data` key holds a (y, x, frame) array; it is transposed to (frame, y, x) on load. The thresholds (`img_thres`, `hist_thres`), the DBSCAN `eps` and the detector geometry (`params`: wavelength, rocking step, distance `D`, `gamma`, `delta`, pixel size, from which `shear_correction` builds the reciprocal-space transform) are also set in `main.py`.

Output is `save_dir/plot_<run>.png` (line profiles) and `save_dir/vtk_<run>.png` (offscreen 3D render). `notebooks/test.ipynb` walks through the same steps with the same hard-coded paths.

## Structure

```
main.py                    line profile extraction, pairing of opposite directions, plots
src/cxd_map/io_util.py     load_roi_mat
src/cxd_map/statistics.py  center_of_mass
src/cxd_map/mapping.py     shear_correction, map2solid_angle, map2spherical, extract_peak_from_2d_map (DBSCAN)
src/cxd_map/visualize.py   vtk_visualize_cxd (3D render and PNG export)
src/cxd_map/analysis.py    extract_peak_directions (not used by main.py)
```

## License

MIT, see [LICENSE](LICENSE).
