import numpy as np
from sklearn.cluster import DBSCAN

def rotation_matrix(angle, axis):
    # Rodrigues' rotation formula implementation
    c, s = np.cos(angle), np.sin(angle)
    ux, uy, uz = axis
    return np.array([
        [c+ux**2*(1-c), ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
        [uy*ux*(1-c)+uz*s, c+uy**2*(1-c), uy*uz*(1-c)-ux*s],
        [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz**2*(1-c)]
    ])

def shear_correction(wavelength: float, delta_theta: float, D: float, gamma: float, delta: float, pixel_size: float) -> np.ndarray:
    s1, s2, s3 = np.eye(3)

    # Negative gamma: the rotation is clockwise at 34-ID-C
    R_delta = rotation_matrix(delta, s2)
    R_gamma = rotation_matrix(gamma, -s1)
    B_det = R_delta @ R_gamma 

    ki_hat = B_det[:, 0]
    kj_hat = B_det[:, 1]

    # |q_i| = |q_j| = pixel_size / (wavelength * D)
    pref = pixel_size / (wavelength * D)
    qi = pref * ki_hat
    qj = pref * kj_hat

    # q_k from the change of q0 over one rocking step
    q0 = (1/wavelength) * (B_det @ s3 - s3)
    dq0 = (rotation_matrix(delta_theta, s2) @ q0) - q0
    qk = -dq0

    B_recip = np.stack([qi, qj, qk], axis=1)

    return B_recip

def map2solid_angle(imgs: np.ndarray, com: np.ndarray, bins: int = 100):
    """
    Map 3D coordinates and intensity to a 2D solid angle representation.
    """
    z_idx, y_idx, x_idx = np.indices(imgs.shape)

    z, y, x = z_idx - com[0], y_idx - com[1], x_idx - com[2]

    r: np.ndarray = np.sqrt(z**2 + y**2 + x**2)
    r: np.ndarray = np.where(r == 0, 1e-10, r)  # Avoid division by zero

    u: np.ndarray = z / r  # u = cos(theta) = z / r [-1, 1]
    phi: np.ndarray = np.arctan2(y, x)

    hist, u_edges, phi_edges = np.histogram2d(
        u.ravel(),
        phi.ravel(),
        bins=[bins, bins],
        range=[[-1, 1], [-np.pi, np.pi]],
        weights=imgs.ravel(),
    )

    return hist, u_edges, phi_edges


def map2spherical(
    imgs: np.ndarray, com: np.ndarray, bins: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Map a 3D image to an intensity-weighted (u=cos(theta), phi) histogram of directions from `com`.
    """
    # indices: (3, Z, Y, X)
    indices = np.indices(imgs.shape).astype(float)
    # com reshaped to (3, 1, 1, 1) for broadcasting
    coords = indices - com[:, None, None, None]

    r = np.linalg.norm(coords, axis=0, keepdims=True)
    r = np.where(r == 0, 1e-10, r)

    unit_vectors = coords / r

    u = unit_vectors[0]  # nz (cos theta)
    phi = np.arctan2(unit_vectors[1], unit_vectors[2])  # ny, nx

    hist, u_edges, phi_edges = np.histogram2d(
        u.ravel(),
        phi.ravel(),
        bins=[bins, bins],
        range=[[-1, 1], [-np.pi, np.pi]],
        weights=imgs.ravel(),
    )

    return unit_vectors, hist, u_edges, phi_edges


def extract_peak_from_2d_map(
    hist: np.ndarray,
    u_edges: np.ndarray,
    phi_edges: np.ndarray,
    eps: float = 0.1,
    min_samples: int = 10,
    threshold_rel: float = 0.2,
) -> np.ndarray:
    """
    Extract 3D peak direction vectors from a 2D (u, phi) histogram by clustering its bright bins.
    """
    u_centers = (u_edges[:-1] + u_edges[1:]) / 2
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
    uu, pp = np.meshgrid(u_centers, phi_centers, indexing="ij")

    mask = hist > (np.max(hist) * threshold_rel)
    u_val = uu[mask]
    p_val = pp[mask]
    w_val = hist[mask]

    if len(u_val) == 0:
        return np.array([])

    # Back to 3D unit vectors, following the map2spherical convention
    # nz = u, ny = r_xy * sin(phi), nx = r_xy * cos(phi)
    r_xy = np.sqrt(np.maximum(0, 1 - u_val**2))
    nz = u_val
    ny = r_xy * np.sin(p_val)
    nx = r_xy * np.cos(p_val)

    v_pts = np.stack([nz, ny, nx], axis=1)  # (N_points, 3)

    # Cluster in 3D vector space, which handles the periodicity of phi
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(
        v_pts, sample_weight=w_val
    )
    labels = clustering.labels_

    peak_vectors = []
    for label in set(labels) - {-1}:
        c_mask = labels == label
        avg_v = np.average(v_pts[c_mask], axis=0, weights=w_val[c_mask])
        peak_vectors.append(avg_v / np.linalg.norm(avg_v))

    return np.array(peak_vectors)


def unit2spherical(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_u = coords[:, 0]
    p_phi = np.arctan2(coords[:, 1], coords[:, 2])
    return p_u, p_phi
