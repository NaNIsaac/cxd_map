# cxd_map

This repository keeps the implementation as it was before it was imported into [CXD](https://github.com/NaNIsaac/CXD) with its history on 2026-10-16 (CXD f669861, moved to `cxd.mapping` in eeb127f). There, `mapping.py` is split into `cxd.mapping.spherical` and `qorageom` (`shear_correction`, `rotation_matrix`), `analysis.py` is `cxd.mapping.analysis`, `visualize.py` is `cxd.mapping.vtk_view`, the line-profile code of `main.py` is `cxd.mapping.profiles` and `scripts/streak_profiles.py`, `load_roi_mat` is `qorageom.io.load_mat` and `center_of_mass` is `cxd.core.statistics.get_com`. `map2solid_angle`, used only in `notebooks/test.ipynb`, was not carried over.

Bragg coherent X-ray diffraction (BCDI) peak analysis: maps a 3D diffraction pattern around its center of mass onto a (cos θ, φ) angular histogram, finds streak/facet directions with DBSCAN, plots intensity line profiles along opposite direction pairs, and renders a 3D VTK view of the pattern in reciprocal-space coordinates.

3D 회절 패턴(ROI `.mat`)의 질량중심 기준 방향 분포에서 스트릭 방향을 찾아 축별 라인 프로파일을 그리는 분석 스크립트. 개인 분석 코드.

## Requirements

Python 3.13+, numpy, scipy, scikit-learn, matplotlib, vtk. `pyproject.toml` 에는 napari, pyvista, trame 도 있지만 코드에서는 쓰지 않음.

## Usage

```powershell
uv sync
$env:PYTHONPATH = "src"     # pyproject 에 build-system 이 없어 cxd_map 이 설치되지 않음
uv run main.py
```

데이터 경로는 환경변수 `DATA` 기준: `main.py` 의 `data_dir`, `save_dir`, `run_ns` (기본 `$DATA\260324_9C\onsite\ROI\scan_0255.mat`) 를 수정. 입력은 `data` 키에 (y, x, frame) 배열이 든 MATLAB `.mat` 파일이며 (frame, y, x) 로 전치해 읽는다. 임계값(`img_thres`, `hist_thres`), DBSCAN `eps`, 검출기 기하(`params`: 파장, 로킹 스텝, 거리 `D`, `gamma`, `delta`, 픽셀 크기 → `shear_correction` 으로 역격자 변환 행렬 계산) 도 `main.py` 에 있다. 결과는 `save_dir/plot_<run>.png` (라인 프로파일) 와 `save_dir/vtk_<run>.png` (3D 렌더, 오프스크린).

`notebooks/test.ipynb` 는 같은 과정을 단계별로 그려 보는 노트북 (경로 동일하게 하드코딩).

## Structure

```
main.py                    라인 프로파일 추출 · 대칭 방향 짝짓기 · 플롯
src/cxd_map/io_util.py     load_roi_mat
src/cxd_map/statistics.py  center_of_mass
src/cxd_map/mapping.py     shear_correction, map2solid_angle, map2spherical, extract_peak_from_2d_map (DBSCAN)
src/cxd_map/visualize.py   vtk_visualize_cxd (3D 렌더 · PNG 저장)
src/cxd_map/analysis.py    extract_peak_directions (main.py 에서는 미사용)
```
