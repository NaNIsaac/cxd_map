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
    Mean intensity profile along `direction` from `com`, averaged over a cube of integer offsets.
    """
    t = np.arange(length)

    offsets = np.arange(-thickness, thickness + 1)
    oz, oy, ox = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    offsets_vec = np.stack(
        [oz.ravel(), oy.ravel(), ox.ravel()], axis=1
    )  # (N_offsets, 3)

    # P(t) = CoM + t*n + offset
    line_points = com + t[:, None] * direction  # (length, 3)
    all_points = (
        line_points[None, :, :] + offsets_vec[:, None, :]
    )  # (N_offsets, length, 3)

    # map_coordinates expects shape (3, N_total_points)
    flat_points = all_points.reshape(-1, 3).T

    # order=1: linear interpolation
    values = map_coordinates(imgs, flat_points, order=1, mode="constant", cval=0.0)

    values = values.reshape(len(offsets_vec), length)
    return np.mean(values, axis=0)


def get_symmetric_profiles(imgs, com, peak_vectors, length=50, thickness=2):
    """
    Pair peaks with opposite directions and return one combined profile per pair.
    """
    num_peaks = len(peak_vectors)
    used = [False] * num_peaks
    combined_results = []

    for i in range(num_peaks):
        if used[i]:
            continue

        # Partner: the unused peak whose dot product is closest to -1
        best_j = -1
        min_dot = 0  # only a negative dot product counts as opposite

        for j in range(num_peaks):
            if i == j or used[j]:
                continue
            dot = np.dot(peak_vectors[i], peak_vectors[j])
            if dot < min_dot:
                min_dot = dot
                best_j = j

        if best_j != -1:
            used[i] = used[best_j] = True

            pos_profile = get_thick_line_profile(
                imgs, com, peak_vectors[i], length, thickness
            )
            neg_profile = get_thick_line_profile(
                imgs, com, peak_vectors[best_j], length, thickness
            )

            # Join into one axis with neg_profile reversed:
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
    # sharex: same x range on every panel so they can be compared
    fig, axes = plt.subplots(num_axes, 1, figsize=(10, 4 * num_axes), sharex=True)

    # plt.subplots returns a bare Axes, not an array, for a single panel
    if num_axes == 1:
        axes = [axes]

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, num_axes))

    for i, (ax, res) in enumerate(zip(axes, combined_results)):
        p1, p2 = res["pair"]

        ax.plot(
            res["dist"],
            res["profile"],
            color=colors[i],
            lw=2.5,
            label=f"Axis {i + 1}: Peaks {p1 + 1} & {p2 + 1}",
        )

        # x=0 is the CoM
        ax.axvline(0, color="white", linestyle="--", alpha=0.3)

        ax.set_title(
            f"Reciprocal Axis {i + 1} (Dot Product: {res['dot']:.4f})", fontsize=12
        )
        ax.set_ylabel("Intensity")
        ax.grid(True, which="both", alpha=0.1)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Distance along axis (pixels)")

    plt.tight_layout()
    plt.show()


def plot_origin_style(combined_results, show=True) -> plt.Figure:
    num_axes = len(combined_results)
    fig, axes = plt.subplots(num_axes, 1, figsize=(8, 4 * num_axes), sharex=True)
    
    if num_axes == 1:
        axes = [axes]
    
    # Tableau10 colors, similar to OriginLab plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

    for i, (ax, res) in enumerate(zip(axes, combined_results)):
        p1, p2 = res["pair"]
        
        ax.plot(res["dist"], res["profile"], color=colors[i % 3], lw=2, 
                label=f"Axis {i+1} (Peaks {p1+1}-{p2+1})")
        
        ax.axvline(0, color="black", lw=1, ls="--", alpha=0.5)

        ax.tick_params(which='both', width=1.5, length=6)
        ax.tick_params(which='minor', length=3)
        
        ax.set_ylabel("Intensity (a.u.)", fontweight='bold')
        ax.legend(frameon=False, loc="upper right") # Origin legends usually have no frame
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
        peaks = extract_peak_from_2d_map(
            hist, u_edges, phi_edges, eps=0.2, threshold_rel=0.3
        )

        q_peaks = []
        for p_idx in peaks:
            q_vec = B_recip @ p_idx 
            q_peaks.append(q_vec / np.linalg.norm(q_vec))
        
        q_peaks = np.array(q_peaks)

        print(f"검출된 피크 개수: {len(peaks)}")
        results = get_symmetric_profiles(imgs, com, peaks, length=50, thickness=3)
        fig = plot_origin_style(results, show=False)

        fig.savefig(save_dir / f"plot_{run_n:04d}.png", dpi=300, bbox_inches='tight')
        vtk_visualize_cxd(imgs, com, results, peaks, B_recip, save_path=save_dir / f"vtk_{run_n:04d}.png", show=False)

if __name__ == "__main__":
    main()
