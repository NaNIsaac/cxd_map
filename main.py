import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cxd_map.io_util import load_roi_mat
from cxd_map.mapping import extract_peak_from_2d_map, map2spherical, shear_correction
from cxd_map.statistics import center_of_mass
from cxd_map.visualize import vtk_visualize_cxd
from scipy.ndimage import gaussian_filter, map_coordinates


def get_thick_line_profile(
    imgs: np.ndarray,
    com: np.ndarray,
    direction: np.ndarray,
    length: int = 100,
    thickness: int = 2,
) -> np.ndarray:
    """
    특정 방향으로 두께를 가진 라인 프로파일을 추출합니다.
    """
    # 1. 선을 따라갈 샘플링 포인트들 (t: 0 ~ length)
    t = np.arange(length)

    # 2. 두께를 만들기 위한 오프셋 생성 (예: 3x3x3 영역 혹은 주변 픽셀)
    # 여기서는 간단하게 중심 주변의 정수 오프셋 범위를 생성합니다.
    offsets = np.arange(-thickness, thickness + 1)
    oz, oy, ox = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    offsets_vec = np.stack(
        [oz.ravel(), oy.ravel(), ox.ravel()], axis=1
    )  # (N_offsets, 3)

    # 3. 모든 샘플링 포인트에 대해 오프셋 적용
    # 결과 구조: (N_offsets, length, 3)
    # P(t) = CoM + t*n + Offset
    line_points = com + t[:, None] * direction  # (length, 3)
    all_points = (
        line_points[None, :, :] + offsets_vec[:, None, :]
    )  # (N_offsets, length, 3)

    # 4. map_coordinates를 위해 차원 재조정 (3, N_total_points)
    flat_points = all_points.reshape(-1, 3).T

    # 5. 3D 볼륨에서 값 추출 (order=1: 선형 보간)
    values = map_coordinates(imgs, flat_points, order=1, mode="constant", cval=0.0)

    # 6. 오프셋별로 다시 나누고 평균 계산
    values = values.reshape(len(offsets_vec), length)
    return np.mean(values, axis=0)


def get_symmetric_profiles(imgs, com, peak_vectors, length=50, thickness=2):
    """
    서로 반대 방향인 피크끼리 짝을 지어 3개의 통합 프로파일을 생성합니다.
    """
    num_peaks = len(peak_vectors)
    used = [False] * num_peaks
    combined_results = []

    for i in range(num_peaks):
        if used[i]:
            continue

        # 현재 피크 i와 가장 '반대'인 피크 j 찾기 (내적이 -1에 가장 가까운 것)
        best_j = -1
        min_dot = 0  # 0보다 작아야 반대 방향

        for j in range(num_peaks):
            if i == j or used[j]:
                continue
            dot = np.dot(peak_vectors[i], peak_vectors[j])
            if dot < min_dot:
                min_dot = dot
                best_j = j

        if best_j != -1:
            used[i] = used[best_j] = True

            # i 방향(+축)과 best_j 방향(-축) 프로파일 추출
            pos_profile = get_thick_line_profile(
                imgs, com, peak_vectors[i], length, thickness
            )
            neg_profile = get_thick_line_profile(
                imgs, com, peak_vectors[best_j], length, thickness
            )

            # neg_profile을 뒤집어서 하나의 축으로 합침
            # [-length, ..., 0, ..., +length]
            full_profile = np.concatenate([neg_profile[::-1], pos_profile[1:]])
            dist_axis = np.linspace(-length, length, len(full_profile))

            combined_results.append(
                {
                    "pair": (i, best_j),
                    "dist": dist_axis,
                    "profile": full_profile,
                    "dot": min_dot,
                }
            )

    return combined_results


def plot_paired_lines_subplots(combined_results):
    num_axes = len(combined_results)
    # 3개의 서브플롯 생성 (세로로 배치)
    # sharex=True: 모든 그래프의 X축 범위를 통일하여 비교를 용이하게 함
    fig, axes = plt.subplots(num_axes, 1, figsize=(10, 4 * num_axes), sharex=True)

    # 만약 결과가 1개뿐일 경우 axes가 배열이 아니게 되는 상황 방지
    if num_axes == 1:
        axes = [axes]

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, num_axes))

    for i, (ax, res) in enumerate(zip(axes, combined_results)):
        p1, p2 = res["pair"]

        # 메인 프로파일 플롯
        ax.plot(
            res["dist"],
            res["profile"],
            color=colors[i],
            lw=2.5,
            label=f"Axis {i + 1}: Peaks {p1 + 1} & {p2 + 1}",
        )

        # 중심선(CoM) 표시
        ax.axvline(0, color="white", linestyle="--", alpha=0.3)

        # 서브플롯별 설정
        ax.set_title(
            f"Reciprocal Axis {i + 1} (Dot Product: {res['dot']:.4f})", fontsize=12
        )
        ax.set_ylabel("Intensity")
        ax.grid(True, which="both", alpha=0.1)
        ax.legend(loc="upper right")

    # 최하단 그래프에만 X축 라벨 표시
    axes[-1].set_xlabel("Distance along axis (pixels)")

    plt.tight_layout()
    plt.show()


def plot_origin_style(combined_results, show=True) -> plt.Figure:
    num_axes = len(combined_results)
    fig, axes = plt.subplots(num_axes, 1, figsize=(8, 4 * num_axes), sharex=True)
    
    if num_axes == 1:
        axes = [axes]
    
    # Origin 느낌의 뚜렷한 컬러맵 (Tableau10 등 추천)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

    for i, (ax, res) in enumerate(zip(axes, combined_results)):
        p1, p2 = res["pair"]
        
        # 1. 메인 데이터 플롯
        ax.plot(res["dist"], res["profile"], color=colors[i % 3], lw=2, 
                label=f"Axis {i+1} (Peaks {p1+1}-{p2+1})")
        
        # 2. 중심 가이드 라인 (Origin은 보통 검은 점선이나 실선 사용)
        ax.axvline(0, color="black", lw=1, ls="--", alpha=0.5)

        # 3. Origin 세부 설정: Ticks 방향 및 테두리 강조
        ax.tick_params(which='both', width=1.5, length=6)
        ax.tick_params(which='minor', length=3) # 보조 tick도 원한다면
        
        ax.set_ylabel("Intensity (a.u.)", fontweight='bold')
        ax.legend(frameon=False, loc="upper right") # Origin은 대개 legend 테두리 없음
        ax.set_title(f"Reciprocal Axis {i+1}", pad=10)

    axes[-1].set_xlabel("Distance from CoM (pixels)", fontweight='bold')
    
    plt.tight_layout()
    if show:
        plt.show()

    return fig

def main():
    data_dir: Path = Path(os.environ["DATA"], r"260324_9C\onsite\ROI")
    save_dir: Path = Path(os.environ["DATA"], r"260324_9C\analysis")
    run_ns: list[int]   = [255]

    params = {
        "wavelength": 12.398 / 5.55 * 1e-10,      # hc/energy(keV) (m)
        "delta_theta": np.radians(0.8 / 120), # Δθ 
        "D": 0.511, # D 
        "gamma": np.radians(0),  # γ 
        "delta": np.radians(69),  # δ 
        "pixel_size": 5.5e-5           # p 
    }
    B_recip = shear_correction(**params)

    for run_n in run_ns:
        data_file: Path = data_dir / f"scan_{run_n:04d}.mat"

        raw_imgs: np.ndarray = load_roi_mat(data_file)

        print(f"Images shape: {raw_imgs.shape}")
        img_thres = 2.5
        hist_thres = 1.5
        imgs = raw_imgs > img_thres
        com = center_of_mass(imgs)
        print(f"Center of Mass: {com}")
        u_vecs, hist, u_edges, phi_edges = map2spherical(imgs, com, bins=150)
        hist = np.where(hist > hist_thres, hist, 0)
        hist = gaussian_filter(hist, sigma=1)
        # 3. 피크 추출 (DBSCAN)
        # eps=0.1은 약 5.7도 이내의 점들을 하나로 묶음
        peaks = extract_peak_from_2d_map(
            hist, u_edges, phi_edges, eps=0.2, threshold_rel=0.3
        )

        q_peaks = []
        for p_idx in peaks:
            # q_vec = B_recip * n [cite: 133]
            q_vec = B_recip @ p_idx 
            q_peaks.append(q_vec / np.linalg.norm(q_vec)) # 단위 벡터화
        
        q_peaks = np.array(q_peaks)

        print(f"검출된 피크 개수: {len(peaks)}")
        results = get_symmetric_profiles(imgs, com, peaks, length=50, thickness=3)
        fig = plot_origin_style(results, show=False)

        fig.savefig(save_dir / f"plot_{run_n:04d}.png", dpi=300, bbox_inches='tight')
        vtk_visualize_cxd(imgs, com, results, peaks, B_recip, save_path=save_dir / f"vtk_{run_n:04d}.png", show=False)

if __name__ == "__main__":
    main()
