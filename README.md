# cxd_map

Superseded by [CXD](https://github.com/NaNIsaac/CXD), which contains this code (`cxd.mapping.spherical`, `cxd.mapping.analysis`, `cxd.mapping.vtk_view` and `scripts/streak_profiles.py`). This repository is no longer maintained; use CXD for new work.

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
