import numba
import concurrent
from functools import partial
from itertools import combinations_with_replacement
from typing import Tuple

import numpy as np
from ase import Atoms
from ase.units import fs
from numpy.typing import NDArray

from dynasor.logging_tools import logger
from dynasor.trajectory import Trajectory, WindowIterator
from dynasor.sample import DynamicSample, StaticSample
from dynasor.post_processing import fourier_cos_filon
from dynasor.core.time_averager import TimeAverager
from dynasor.core.reciprocal import calc_rho_q, calc_rho_j_q
from dynasor.qpoints.tools import get_index_offset
from dynasor.units import radians_per_fs_to_meV

def compute_dynamic_structure_factors_wk_currents(
    traj: Trajectory,
    q_points: NDArray[float],
    dt: float,
    window_size: int,
    window_step: int = 1,
    calculate_currents: bool = False,
    calculate_incoherent: bool = False,
    currents_4group: bool = False,
    # --- W-K options ---
    wk_window: str = "hann",   # "hann" or "rect"
    wk_detrend: bool = True,   # remove mean from j(t) inside each window (recommended)
) -> DynamicSample:
    """
    Same overall structure as dynasor's compute_dynamic_structure_factors, but
    current spectra (Clqw/Ctqw/Ct1qw/Ct2qw) are computed by Wiener–Khinchin / Welch:
        - build j(q,t) time series inside each window
        - FFT along t -> J(q,omega)
        - spectrum per window:
              auto:  J_s1 * conj(J_s2)   (pairs supported)
              diag-only 4group: sum_g J_{s1,g} * conj(J_{s2,g})   (no cross-group terms)
        - average over windows

    Density correlations F(q,t) and S(q,omega) remain computed via the original time-correlation route.
    """

    # sanity checks
    if q_points.shape[1] != 3:
        raise ValueError("q-points array has the wrong shape.")
    if dt <= 0:
        raise ValueError(f"dt must be positive: dt={dt}")
    if window_size <= 2:
        raise ValueError(f"window_size must be larger than 2: window_size={window_size}")
    if window_size % 2 != 0:
        raise ValueError(f"window_size must be even: window_size={window_size}")
    if window_step <= 0:
        raise ValueError(f"window_step must be positive: window_step={window_step}")
    if currents_4group and (not calculate_currents):
        raise ValueError("currents_4group=True requires calculate_currents=True")

    # internal parameters
    n_qpoints = q_points.shape[0]
    delta_t = traj.frame_step * dt
    N_tc = window_size + 1  # number of time samples inside each window

    # frequency grid: match dynasor convention (N_tc points from Filon, but here FFT gives rfft bins)
    # We'll store omega_rfft (length Nw) in data_dict_corr['omega'] for current spectra.
    freq = np.fft.rfftfreq(N_tc, d=delta_t)     # cycles/fs
    omega_rfft = 2 * np.pi * freq               # rad/fs
    Nw = omega_rfft.size

    # logging similar to original
    dw = np.pi / (window_size * delta_t)
    w_max = dw * window_size
    w_N = 2 * np.pi / (2 * delta_t)
    logger.info(f"Spacing between samples (frame_step): {traj.frame_step}")
    logger.info(f"Time between consecutive frames in input trajectory (dt): {dt} fs")
    logger.info(f"Time between consecutive frames used (dt * frame_step): {delta_t} fs")
    logger.info(f"Time window size (dt * frame_step * window_size): {delta_t * window_size:.1f} fs")
    logger.info(f"Angular frequency resolution (Filon-style): dw = {dw:.6f} rad/fs = {dw * radians_per_fs_to_meV:.3f} meV")
    logger.info(f"Max angular frequency (Filon-style): {w_max:.6f} rad/fs = {w_max * radians_per_fs_to_meV:.3f} meV")
    logger.info(f"Nyquist angular frequency: {w_N:.6f} rad/fs = {w_N * radians_per_fs_to_meV:.3f} meV")

    if calculate_currents:
        logger.info("Calculating current spectra using windowed Wiener–Khinchin (Welch)")
        logger.info(f"  FFT bins: N_tc={N_tc} -> Nw(rfft)={Nw}")
        logger.info(f"  window: {wk_window}, detrend: {wk_detrend}")
        if currents_4group:
            logger.info("  Using MATLAB-style 4-group partition for currents (sum over groups; no cross-group terms)")
    if calculate_incoherent:
        logger.info("Calculating incoherent part (self-part) of correlations")

    logger.info(f"Number of q-points: {n_qpoints}")

    # q unit vectors
    q_directions = q_points.copy()
    q_distances = np.linalg.norm(q_points, axis=1)
    nonzero = q_distances > 0
    q_directions[nonzero] /= q_distances[nonzero].reshape(-1, 1)

    # fixed transverse basis (assumes q || z)
    e1 = np.tile(np.array([1.0, 0.0, 0.0]), (n_qpoints, 1))
    e2 = np.tile(np.array([0.0, 1.0, 0.0]), (n_qpoints, 1))

    # FFT window weights
    if wk_window.lower() == "hann":
        w_t = np.hanning(N_tc).astype(float)
    elif wk_window.lower() in ("rect", "boxcar", "none"):
        w_t = np.ones((N_tc,), dtype=float)
    else:
        raise ValueError("wk_window must be 'hann' or 'rect'")

    # --- 4-group partition indices (per atom_type) ---
    group_count = 4
    group_local_idx = None
    if currents_4group:
        group_local_idx = [dict() for _ in range(group_count)]
        for atom_type in traj.atom_types:
            gidx = np.asarray(traj.atomic_indices[atom_type], dtype=np.int64)  # global indices
            local = np.arange(len(gidx), dtype=np.int64)
            for g in range(group_count):
                group_local_idx[g][atom_type] = local[(gidx % group_count) == g]

    # --- element processors ---
    def f2_rho(frame):
        rho_qs_dict = {}
        for atom_type in frame.positions_by_type.keys():
            x = frame.positions_by_type[atom_type]
            rho_qs_dict[atom_type] = calc_rho_q(x, q_points)
        frame.rho_qs_dict = rho_qs_dict
        return frame

    def f2_rho_and_j(frame):
        rho_qs_dict = {}

        # store j-components needed for WK inside each frame
        if not currents_4group:
            jz_qs_dict = {}
            jt1_qs_dict = {}
            jt2_qs_dict = {}
            jper_qs_dict = {}
        else:
            jz_qs_dict_g  = [dict() for _ in range(group_count)]
            jt1_qs_dict_g = [dict() for _ in range(group_count)]
            jt2_qs_dict_g = [dict() for _ in range(group_count)]
            jper_qs_dict_g = [dict() for _ in range(group_count)]

        for atom_type in frame.positions_by_type.keys():
            x_all = frame.positions_by_type[atom_type]
            v_all = frame.velocities_by_type[atom_type]

            rho_qs_dict[atom_type] = calc_rho_q(x_all, q_points)

            if not currents_4group:
                _, j_qs = calc_rho_j_q(x_all, v_all, q_points)  # (n_q,3)
                jz = np.sum(j_qs * q_directions, axis=1)        # (n_q,)
                jper = j_qs - (jz[:, None] * q_directions)      # (n_q,3)
                jt1 = np.sum(j_qs * e1, axis=1)                 # (n_q,)
                jt2 = np.sum(j_qs * e2, axis=1)                 # (n_q,)
                jz_qs_dict[atom_type] = jz
                jper_qs_dict[atom_type] = jper
                jt1_qs_dict[atom_type] = jt1
                jt2_qs_dict[atom_type] = jt2
            else:
                for g in range(group_count):
                    sel = group_local_idx[g][atom_type]
                    if sel.size == 0:
                        jz_qs_dict_g[g][atom_type] = np.zeros((n_qpoints,), dtype=np.complex128)
                        jper_qs_dict_g[g][atom_type] = np.zeros((n_qpoints, 3), dtype=np.complex128)
                        jt1_qs_dict_g[g][atom_type] = np.zeros((n_qpoints,), dtype=np.complex128)
                        jt2_qs_dict_g[g][atom_type] = np.zeros((n_qpoints,), dtype=np.complex128)
                        continue
                    x = x_all[sel]
                    v = v_all[sel]
                    _, j_qs = calc_rho_j_q(x, v, q_points)
                    jz = np.sum(j_qs * q_directions, axis=1)
                    jper = j_qs - (jz[:, None] * q_directions)
                    jt1 = np.sum(j_qs * e1, axis=1)
                    jt2 = np.sum(j_qs * e2, axis=1)
                    jz_qs_dict_g[g][atom_type] = jz
                    jper_qs_dict_g[g][atom_type] = jper
                    jt1_qs_dict_g[g][atom_type] = jt1
                    jt2_qs_dict_g[g][atom_type] = jt2

        frame.rho_qs_dict = rho_qs_dict
        if not currents_4group:
            frame.jz_qs_dict = jz_qs_dict
            frame.jper_qs_dict = jper_qs_dict
            frame.jt1_qs_dict = jt1_qs_dict
            frame.jt2_qs_dict = jt2_qs_dict
        else:
            frame.jz_qs_dict_g = jz_qs_dict_g
            frame.jper_qs_dict_g = jper_qs_dict_g
            frame.jt1_qs_dict_g = jt1_qs_dict_g
            frame.jt2_qs_dict_g = jt2_qs_dict_g

        return frame

    element_processor = f2_rho_and_j if calculate_currents else f2_rho

    # window iterator
    window_iterator = WindowIterator(
        traj,
        width=N_tc,
        window_step=window_step,
        element_processor=element_processor
    )

    # pairs
    pairs = list(combinations_with_replacement(traj.atom_types, r=2))
    particle_counts = {key: len(val) for key, val in traj.atomic_indices.items()}

    # --- density time-correlation averagers (keep original) ---
    F_q_t_averager = {pair: TimeAverager(N_tc, n_qpoints) for pair in pairs}
    if calculate_incoherent:
        F_s_q_t_averager = {atom_type: TimeAverager(N_tc, n_qpoints) for atom_type in traj.atom_types}

    # --- current spectral accumulators (WK) ---
    if calculate_currents:
        Clqw_acc  = {pair: np.zeros((n_qpoints, Nw), dtype=np.complex128) for pair in pairs}
        Ctqw_acc  = {pair: np.zeros((n_qpoints, Nw), dtype=np.complex128) for pair in pairs}
        Ct1qw_acc = {pair: np.zeros((n_qpoints, Nw), dtype=np.complex128) for pair in pairs}
        Ct2qw_acc = {pair: np.zeros((n_qpoints, Nw), dtype=np.complex128) for pair in pairs}
        nwin = 0

    # correlation function for density & incoh only (current handled per-window)
    def calc_corr_density(window, time_i):
        f0 = window[0]
        fi = window[time_i]
        for s1, s2 in pairs:
            Fqt = np.real(f0.rho_qs_dict[s1] * fi.rho_qs_dict[s2].conjugate())
            if s1 != s2:
                Fqt += np.real(f0.rho_qs_dict[s2] * fi.rho_qs_dict[s1].conjugate())
            F_q_t_averager[(s1, s2)].add_sample(time_i, Fqt)

        if calculate_incoherent:
            for atom_type in traj.atom_types:
                xi = fi.positions_by_type[atom_type]
                x0 = f0.positions_by_type[atom_type]
                Fsqt = np.real(calc_rho_q(xi - x0, q_points))
                F_s_q_t_averager[atom_type].add_sample(time_i, Fsqt)

    # main loop
    logging_interval = 1000
    with concurrent.futures.ThreadPoolExecutor() as tpe:
        for window in window_iterator:
            if window[0].frame_index % logging_interval == 0:
                logger.info(f"Processing window {window[0].frame_index} to {window[-1].frame_index}")

            # ---- density/incoh time-domain averaging (same as original) ----
            for _ in tpe.map(partial(calc_corr_density, window), range(len(window))):
                pass

            # ---- current spectra via WK/Welch ----
            if calculate_currents:
                # build time series per atom_type (and per group if enabled)
                if not currents_4group:
                    # arrays: (N_tc, n_qpoints) complex
                    jz_ts  = {s: np.zeros((N_tc, n_qpoints), dtype=np.complex128) for s in traj.atom_types}
                    jt1_ts = {s: np.zeros((N_tc, n_qpoints), dtype=np.complex128) for s in traj.atom_types}
                    jt2_ts = {s: np.zeros((N_tc, n_qpoints), dtype=np.complex128) for s in traj.atom_types}
                    jper_ts = {s: np.zeros((N_tc, n_qpoints, 3), dtype=np.complex128) for s in traj.atom_types}

                    for ti, fr in enumerate(window):
                        for s in traj.atom_types:
                            jz_ts[s][ti]  = fr.jz_qs_dict[s]
                            jt1_ts[s][ti] = fr.jt1_qs_dict[s]
                            jt2_ts[s][ti] = fr.jt2_qs_dict[s]
                            jper_ts[s][ti] = fr.jper_qs_dict[s]

                    # windowing + detrend
                    wt = w_t[:, None]
                    for s in traj.atom_types:
                        if wk_detrend:
                            jz_ts[s]  = jz_ts[s]  - np.mean(jz_ts[s],  axis=0, keepdims=True)
                            jt1_ts[s] = jt1_ts[s] - np.mean(jt1_ts[s], axis=0, keepdims=True)
                            jt2_ts[s] = jt2_ts[s] - np.mean(jt2_ts[s], axis=0, keepdims=True)
                            jper_ts[s] = jper_ts[s] - np.mean(jper_ts[s], axis=0, keepdims=True)

                        jz_ts[s]  = jz_ts[s]  * wt
                        jt1_ts[s] = jt1_ts[s] * wt
                        jt2_ts[s] = jt2_ts[s] * wt
                        jper_ts[s] = jper_ts[s] * wt[:, :, None]

                    # FFT along time -> (Nw, n_qpoints)
                    Jz  = {s: np.fft.rfft(jz_ts[s],  axis=0) for s in traj.atom_types}
                    Jt1 = {s: np.fft.rfft(jt1_ts[s], axis=0) for s in traj.atom_types}
                    Jt2 = {s: np.fft.rfft(jt2_ts[s], axis=0) for s in traj.atom_types}
                    Jper = {s: np.fft.rfft(jper_ts[s], axis=0) for s in traj.atom_types}  # (Nw,n_q,3)

                    # accumulate spectra per pair: J_s1 * conj(J_s2)
                    for s1, s2 in pairs:
                        # longitudinal
                        Cl = (Jz[s1] * np.conjugate(Jz[s2])) / N_tc          # (Nw,n_q)
                        # transverse vector: 0.5*sum(Jper*conj(Jper))
                        Ct = 0.5 * np.sum(Jper[s1] * np.conjugate(Jper[s2]), axis=2) / N_tc  # (Nw,n_q)
                        # split transverse
                        Ct1 = (Jt1[s1] * np.conjugate(Jt1[s2])) / N_tc
                        Ct2 = (Jt2[s1] * np.conjugate(Jt2[s2])) / N_tc

                        # symmetrize for s1!=s2 to keep real part consistent with original double-count
                        if s1 != s2:
                            Cl += (Jz[s2] * np.conjugate(Jz[s1])) / N_tc
                            Ct += 0.5 * np.sum(Jper[s2] * np.conjugate(Jper[s1]), axis=2) / N_tc
                            Ct1 += (Jt1[s2] * np.conjugate(Jt1[s1])) / N_tc
                            Ct2 += (Jt2[s2] * np.conjugate(Jt2[s1])) / N_tc

                        # store as (n_q, Nw)
                        Clqw_acc[(s1, s2)]  += Cl.T
                        Ctqw_acc[(s1, s2)]  += Ct.T
                        Ct1qw_acc[(s1, s2)] += Ct1.T
                        Ct2qw_acc[(s1, s2)] += Ct2.T

                else:
                    # 4-group diag-only: sum_g J_{g} * conj(J_{g})
                    # build per group time series first
                    Jz_g  = {s: [] for s in traj.atom_types}
                    Jt1_g = {s: [] for s in traj.atom_types}
                    Jt2_g = {s: [] for s in traj.atom_types}
                    Jper_g = {s: [] for s in traj.atom_types}

                    for g in range(group_count):
                        # per group arrays
                        jz_ts  = {s: np.zeros((N_tc, n_qpoints), dtype=np.complex128) for s in traj.atom_types}
                        jt1_ts = {s: np.zeros((N_tc, n_qpoints), dtype=np.complex128) for s in traj.atom_types}
                        jt2_ts = {s: np.zeros((N_tc, n_qpoints), dtype=np.complex128) for s in traj.atom_types}
                        jper_ts = {s: np.zeros((N_tc, n_qpoints, 3), dtype=np.complex128) for s in traj.atom_types}

                        for ti, fr in enumerate(window):
                            for s in traj.atom_types:
                                jz_ts[s][ti]   = fr.jz_qs_dict_g[g][s]
                                jt1_ts[s][ti]  = fr.jt1_qs_dict_g[g][s]
                                jt2_ts[s][ti]  = fr.jt2_qs_dict_g[g][s]
                                jper_ts[s][ti] = fr.jper_qs_dict_g[g][s]

                        wt = w_t[:, None]
                        for s in traj.atom_types:
                            if wk_detrend:
                                jz_ts[s]  = jz_ts[s]  - np.mean(jz_ts[s],  axis=0, keepdims=True)
                                jt1_ts[s] = jt1_ts[s] - np.mean(jt1_ts[s], axis=0, keepdims=True)
                                jt2_ts[s] = jt2_ts[s] - np.mean(jt2_ts[s], axis=0, keepdims=True)
                                jper_ts[s] = jper_ts[s] - np.mean(jper_ts[s], axis=0, keepdims=True)

                            jz_ts[s]  *= wt
                            jt1_ts[s] *= wt
                            jt2_ts[s] *= wt
                            jper_ts[s] *= wt[:, :, None]

                        # FFT
                        for s in traj.atom_types:
                            Jz_g[s].append(np.fft.rfft(jz_ts[s], axis=0))        # (Nw,n_q)
                            Jt1_g[s].append(np.fft.rfft(jt1_ts[s], axis=0))
                            Jt2_g[s].append(np.fft.rfft(jt2_ts[s], axis=0))
                            Jper_g[s].append(np.fft.rfft(jper_ts[s], axis=0))    # (Nw,n_q,3)

                    # accumulate per pair: sum_g (J_g(s1) * conj(J_g(s2)))  (no cross-g terms)
                    for s1, s2 in pairs:
                        Cl_sum = np.zeros((Nw, n_qpoints), dtype=np.complex128)
                        Ct_sum = np.zeros((Nw, n_qpoints), dtype=np.complex128)
                        Ct1_sum = np.zeros((Nw, n_qpoints), dtype=np.complex128)
                        Ct2_sum = np.zeros((Nw, n_qpoints), dtype=np.complex128)

                        for g in range(group_count):
                            Cl_sum += (Jz_g[s1][g] * np.conjugate(Jz_g[s2][g])) / N_tc
                            Ct_sum += 0.5 * np.sum(Jper_g[s1][g] * np.conjugate(Jper_g[s2][g]), axis=2) / N_tc
                            Ct1_sum += (Jt1_g[s1][g] * np.conjugate(Jt1_g[s2][g])) / N_tc
                            Ct2_sum += (Jt2_g[s1][g] * np.conjugate(Jt2_g[s2][g])) / N_tc

                            if s1 != s2:
                                Cl_sum += (Jz_g[s2][g] * np.conjugate(Jz_g[s1][g])) / N_tc
                                Ct_sum += 0.5 * np.sum(Jper_g[s2][g] * np.conjugate(Jper_g[s1][g]), axis=2) / N_tc
                                Ct1_sum += (Jt1_g[s2][g] * np.conjugate(Jt1_g[s1][g])) / N_tc
                                Ct2_sum += (Jt2_g[s2][g] * np.conjugate(Jt2_g[s1][g])) / N_tc

                        Clqw_acc[(s1, s2)]  += Cl_sum.T
                        Ctqw_acc[(s1, s2)]  += Ct_sum.T
                        Ct1qw_acc[(s1, s2)] += Ct1_sum.T
                        Ct2qw_acc[(s1, s2)] += Ct2_sum.T

                nwin += 1

    # ---- collect results dict (keep density part identical) ----
    data_dict_corr = {}
    time = delta_t * np.arange(N_tc, dtype=float)
    data_dict_corr["q_points"] = q_points
    data_dict_corr["time"] = time

    # density spectra (unchanged)
    F_q_t_tot = np.zeros((n_qpoints, N_tc))
    S_q_w_tot = np.zeros((n_qpoints, N_tc))
    for pair in pairs:
        key = "_".join(pair)
        F_q_t = 1 / traj.n_atoms * F_q_t_averager[pair].get_average_all()
        w_filon, S_q_w = fourier_cos_filon(F_q_t, delta_t)
        S_q_w = np.array(S_q_w)
        data_dict_corr["omega_filon"] = w_filon  # keep original omega under different key
        data_dict_corr[f"Fqt_coh_{key}"] = F_q_t
        data_dict_corr[f"Sqw_coh_{key}"] = S_q_w
        F_q_t_tot += F_q_t
        S_q_w_tot += S_q_w
    data_dict_corr["Fqt_coh"] = F_q_t_tot
    data_dict_corr["Sqw_coh"] = S_q_w_tot

    # current spectra (WK)
    if calculate_currents:
        if nwin == 0:
            raise RuntimeError("No windows processed for WK currents. Check window_size/window_step.")

        data_dict_corr["omega"] = omega_rfft  # <-- current omega (rad/fs)

        Clqw_tot  = np.zeros((n_qpoints, Nw), dtype=np.float64)
        Ctqw_tot  = np.zeros((n_qpoints, Nw), dtype=np.float64)
        Ct1qw_tot = np.zeros((n_qpoints, Nw), dtype=np.float64)
        Ct2qw_tot = np.zeros((n_qpoints, Nw), dtype=np.float64)

        for pair in pairs:
            key = "_".join(pair)

            Clqw = (Clqw_acc[pair] / nwin) / traj.n_atoms
            Ctqw = (Ctqw_acc[pair] / nwin) / traj.n_atoms
            Ct1qw = (Ct1qw_acc[pair] / nwin) / traj.n_atoms
            Ct2qw = (Ct2qw_acc[pair] / nwin) / traj.n_atoms

            # store real parts (symmetrized already)
            Clqw_r = np.real(Clqw)
            Ctqw_r = np.real(Ctqw)
            Ct1qw_r = np.real(Ct1qw)
            Ct2qw_r = np.real(Ct2qw)

            data_dict_corr[f"Clqw_{key}"]  = Clqw_r
            data_dict_corr[f"Ctqw_{key}"]  = Ctqw_r
            data_dict_corr[f"Ct1qw_{key}"] = Ct1qw_r
            data_dict_corr[f"Ct2qw_{key}"] = Ct2qw_r

            Clqw_tot  += Clqw_r
            Ctqw_tot  += Ctqw_r
            Ct1qw_tot += Ct1qw_r
            Ct2qw_tot += Ct2qw_r

        data_dict_corr["Clqw"]  = Clqw_tot
        data_dict_corr["Ctqw"]  = Ctqw_tot
        data_dict_corr["Ct1qw"] = Ct1qw_tot
        data_dict_corr["Ct2qw"] = Ct2qw_tot

    # incoherent spectra (unchanged, still filon)
    if calculate_incoherent:
        Fs_q_t_tot = np.zeros((n_qpoints, N_tc))
        Ss_q_w_tot = np.zeros((n_qpoints, N_tc))
        for atom_type in traj.atom_types:
            Fs_q_t = 1 / traj.n_atoms * F_s_q_t_averager[atom_type].get_average_all()
            _, Ss_q_w = fourier_cos_filon(Fs_q_t, delta_t)
            data_dict_corr[f"Fqt_incoh_{atom_type}"] = Fs_q_t
            data_dict_corr[f"Sqw_incoh_{atom_type}"] = Ss_q_w
            Fs_q_t_tot += Fs_q_t
            Ss_q_w_tot += Ss_q_w

        data_dict_corr["Fqt_incoh"] = Fs_q_t_tot
        data_dict_corr["Sqw_incoh"] = Ss_q_w_tot
        data_dict_corr["Fqt"] = data_dict_corr["Fqt_coh"] + data_dict_corr["Fqt_incoh"]
        data_dict_corr["Sqw"] = data_dict_corr["Sqw_coh"] + data_dict_corr["Sqw_incoh"]
    else:
        data_dict_corr["Fqt"] = data_dict_corr["Fqt_coh"].copy()
        data_dict_corr["Sqw"] = data_dict_corr["Sqw_coh"].copy()

    # finalize (keep meta)
    result = DynamicSample(
        data_dict_corr,
        atom_types=traj.atom_types,
        pairs=pairs,
        particle_counts=particle_counts,
        cell=traj.cell,
        time_between_frames=delta_t,
        maximum_time_lag=delta_t * window_size,
        angular_frequency_resolution=dw,
        maximum_angular_frequency=w_max,
        number_of_frames=traj.number_of_frames_read,
    )
    return result

def currents_wk_windowed(
    traj,
    q_points,
    dt,
    window_size,
    window_step=1,
    use_hann=True,
):
    """
    Windowed Wiener–Khinchin / Welch-style estimator for current spectra.
    Returns:
      omega (rad/fs), Clqw(q,omega), Ctqw(q,omega)
    Shapes:
      Clqw: (n_qpoints, n_omega)
    """
    n_qpoints = q_points.shape[0]
    delta_t = traj.frame_step * dt
    N = window_size + 1  # dynasor의 N_tc와 맞추기

    # q unit vectors
    q_dir = q_points.copy()
    q_norm = np.linalg.norm(q_points, axis=1)
    nz = q_norm > 0
    q_dir[nz] /= q_norm[nz][:, None]

    # window function
    if use_hann:
        w = np.hanning(N).astype(float)
    else:
        w = np.ones(N, dtype=float)

    # rfft frequency grid (cycles/fs) -> omega(rad/fs)
    freq = np.fft.rfftfreq(N, d=delta_t)   # 1/fs
    omega = 2*np.pi*freq                   # rad/fs
    n_omega = omega.size

    # 누적용 (윈도우 평균)
    Clqw_acc = np.zeros((n_qpoints, n_omega), dtype=float)
    Ctqw_acc = np.zeros((n_qpoints, n_omega), dtype=float)
    nwin = 0

    # 윈도우 iterator: 프레임별로 j(q,t)만 뽑아오면 됨
    def proc_frame(frame):
        # atom_types가 1개('X')라고 가정 (필요시 확장 가능)
        atom_type = traj.atom_types[0]
        x = frame.positions_by_type[atom_type]
        v = frame.velocities_by_type[atom_type]
        _, j_q = calc_rho_j_q(x, v, q_points)   # (n_q,3) complex
        jL = np.sum(j_q * q_dir, axis=1)        # (n_q,) complex
        jT_vec = j_q - jL[:, None]*q_dir        # (n_q,3)
        return jL, jT_vec

    window_it = WindowIterator(
        traj,
        width=N,
        window_step=window_step,
        element_processor=None
    )

    for window in window_it:
        # window: list[TrajectoryFrame] length N
        # 시계열 배열로 쌓기
        jL_t = np.zeros((N, n_qpoints), dtype=np.complex128)
        jT_t = np.zeros((N, n_qpoints, 3), dtype=np.complex128)

        for ti, frame in enumerate(window):
            jL, jTvec = proc_frame(frame)
            jL_t[ti] = jL
            jT_t[ti] = jTvec

        # windowing
        jL_t_w = jL_t * w[:, None]
        jT_t_w = jT_t * w[:, None, None]

        # FFT along time axis
        JL = np.fft.rfft(jL_t_w, axis=0)               # (n_omega, n_q)
        JT = np.fft.rfft(jT_t_w, axis=0)               # (n_omega, n_q, 3)

        # Power spectra (W-K/Welch)
        # longitudinal: |JL|^2
        Cl = (JL * JL.conjugate()).real                # (n_omega, n_q)

        # transverse: 0.5 * |JT|^2 summed over xyz
        Ct = 0.5 * (np.sum(JT * JT.conjugate(), axis=2).real)  # (n_omega, n_q)

        # accumulate (transpose to match dynasor convention (n_q, n_omega))
        Clqw_acc += Cl.T
        Ctqw_acc += Ct.T
        nwin += 1

    if nwin == 0:
        raise RuntimeError("No windows processed. Check window_size/window_step/frame range.")

    # 평균
    Clqw = Clqw_acc / nwin / traj.n_atoms
    Ctqw = Ctqw_acc / nwin / traj.n_atoms

    return omega, Clqw, Ctqw

def compute_dynamic_structure_factors(
    traj: Trajectory,
    q_points: NDArray[float],
    dt: float,
    window_size: int,
    window_step: int = 1,
    calculate_currents: bool = False,
    calculate_incoherent: bool = False,
    # --- MATLAB-style option ---
    currents_4group: bool = False,
) -> DynamicSample:
    """
    NOTE:
      - This version additionally computes transverse spectra split into two fixed polarizations:
          Ct1 (e1 = x̂) and Ct2 (e2 = ŷ)
      - This is only physically correct if your q-path is along z (Γ→Z), so that x̂ and ŷ are ⟂ q.
    """

    # sanity check input args
    if q_points.shape[1] != 3:
        raise ValueError('q-points array has the wrong shape.')
    if dt <= 0:
        raise ValueError(f'dt must be positive: dt= {dt}')
    if window_size <= 2:
        raise ValueError(f'window_size must be larger than 2: window_size= {window_size}')
    if window_size % 2 != 0:
        raise ValueError(f'window_size must be even: window_size= {window_size}')
    if window_step <= 0:
        raise ValueError(f'window_step must be positive: window_step= {window_step}')
    if currents_4group and (not calculate_currents):
        raise ValueError("currents_4group=True requires calculate_currents=True")

    # define internal parameters
    n_qpoints = q_points.shape[0]
    delta_t = traj.frame_step * dt
    N_tc = window_size + 1

    # log all setup information
    dw = np.pi / (window_size * delta_t)
    w_max = dw * window_size
    w_N = 2 * np.pi / (2 * delta_t)  # Nyquist angular frequency

    logger.info(f'Spacing between samples (frame_step): {traj.frame_step}')
    logger.info(f'Time between consecutive frames in input trajectory (dt): {dt} fs')
    logger.info(f'Time between consecutive frames used (dt * frame_step): {delta_t} fs')
    logger.info(f'Time window size (dt * frame_step * window_size): {delta_t * window_size:.1f} fs')
    logger.info(f'Angular frequency resolution: dw = {dw:.6f} rad/fs = '
                f'{dw * radians_per_fs_to_meV:.3f} meV')
    logger.info(f'Maximum angular frequency (dw * window_size):'
                f' {w_max:.6f} rad/fs = {w_max * radians_per_fs_to_meV:.3f} meV')
    logger.info(f'Nyquist angular frequency (2pi / frame_step / dt / 2):'
                f' {w_N:.6f} rad/fs = {w_N * radians_per_fs_to_meV:.3f} meV')

    if calculate_currents:
        logger.info('Calculating current (velocity) correlations')
        logger.info('  Also computing Ct1/Ct2 with fixed transverse basis e1=xhat, e2=yhat (valid for q||z)')
        if currents_4group:
            logger.info('  Using MATLAB-style 4-group partition for current correlations (sum over groups; no cross terms)')
    if calculate_incoherent:
        logger.info('Calculating incoherent part (self-part) of correlations')

    # log some info regarding q-points
    logger.info(f'Number of q-points: {n_qpoints}')

    # q unit vectors
    q_directions = q_points.copy()
    q_distances = np.linalg.norm(q_points, axis=1)
    nonzero = q_distances > 0
    q_directions[nonzero] /= q_distances[nonzero].reshape(-1, 1)

    # ---- FIXED transverse basis (for q || z path): e1=xhat, e2=yhat ----
    e1 = np.tile(np.array([1.0, 0.0, 0.0]), (n_qpoints, 1))
    e2 = np.tile(np.array([0.0, 1.0, 0.0]), (n_qpoints, 1))

    # --- MATLAB-style 4-group partition: per atom_type, choose atoms by (global_index % 4) ---
    group_count = 4
    group_local_idx = None
    if currents_4group:
        group_local_idx = [dict() for _ in range(group_count)]
        for atom_type in traj.atom_types:
            gidx = np.asarray(traj.atomic_indices[atom_type], dtype=np.int64)
            local = np.arange(len(gidx), dtype=np.int64)
            for g in range(group_count):
                group_local_idx[g][atom_type] = local[(gidx % group_count) == g]

    # setup functions to process frames
    def f2_rho(frame):
        rho_qs_dict = dict()
        for atom_type in frame.positions_by_type.keys():
            x = frame.positions_by_type[atom_type]
            rho_qs_dict[atom_type] = calc_rho_q(x, q_points)
        frame.rho_qs_dict = rho_qs_dict
        return frame

    def f2_rho_and_j(frame):
        rho_qs_dict = dict()

        # default (non-4group) stores
        jz_qs_dict = dict()
        jper_qs_dict = dict()
        jt1_qs_dict = dict()   # NEW: j·e1
        jt2_qs_dict = dict()   # NEW: j·e2

        # 4-group stores
        if currents_4group:
            jz_qs_dict_g  = [dict() for _ in range(group_count)]
            jper_qs_dict_g = [dict() for _ in range(group_count)]
            jt1_qs_dict_g = [dict() for _ in range(group_count)]  # NEW
            jt2_qs_dict_g = [dict() for _ in range(group_count)]  # NEW

        for atom_type in frame.positions_by_type.keys():
            x_all = frame.positions_by_type[atom_type]
            v_all = frame.velocities_by_type[atom_type]

            # Density correlations: unchanged
            rho_qs_dict[atom_type] = calc_rho_q(x_all, q_points)

            if not currents_4group:
                _, j_qs = calc_rho_j_q(x_all, v_all, q_points)  # (#q,3)

                # longitudinal / transverse-vector
                jz_qs = np.sum(j_qs * q_directions, axis=1)             # (#q,)
                jper_qs = j_qs - (jz_qs[:, None] * q_directions)        # (#q,3)

                # NEW: split transverse into fixed e1/e2 components
                jt1_qs = np.sum(j_qs * e1, axis=1)                      # (#q,)
                jt2_qs = np.sum(j_qs * e2, axis=1)                      # (#q,)

                jz_qs_dict[atom_type] = jz_qs
                jper_qs_dict[atom_type] = jper_qs
                jt1_qs_dict[atom_type] = jt1_qs
                jt2_qs_dict[atom_type] = jt2_qs
            else:
                for g in range(group_count):
                    sel = group_local_idx[g][atom_type]
                    if sel.size == 0:
                        jz_qs_dict_g[g][atom_type]   = np.zeros((n_qpoints,), dtype=np.complex128)
                        jper_qs_dict_g[g][atom_type] = np.zeros((n_qpoints, 3), dtype=np.complex128)
                        jt1_qs_dict_g[g][atom_type]  = np.zeros((n_qpoints,), dtype=np.complex128)  # NEW
                        jt2_qs_dict_g[g][atom_type]  = np.zeros((n_qpoints,), dtype=np.complex128)  # NEW
                        continue

                    x = x_all[sel]
                    v = v_all[sel]
                    _, j_qs = calc_rho_j_q(x, v, q_points)

                    jz_qs = np.sum(j_qs * q_directions, axis=1)
                    jper_qs = j_qs - (jz_qs[:, None] * q_directions)

                    # NEW
                    jt1_qs = np.sum(j_qs * e1, axis=1)
                    jt2_qs = np.sum(j_qs * e2, axis=1)

                    jz_qs_dict_g[g][atom_type]   = jz_qs
                    jper_qs_dict_g[g][atom_type] = jper_qs
                    jt1_qs_dict_g[g][atom_type]  = jt1_qs
                    jt2_qs_dict_g[g][atom_type]  = jt2_qs

        frame.rho_qs_dict = rho_qs_dict

        if not currents_4group:
            frame.jz_qs_dict = jz_qs_dict
            frame.jper_qs_dict = jper_qs_dict
            frame.jt1_qs_dict = jt1_qs_dict   # NEW
            frame.jt2_qs_dict = jt2_qs_dict   # NEW
        else:
            frame.jz_qs_dict_g = jz_qs_dict_g
            frame.jper_qs_dict_g = jper_qs_dict_g
            frame.jt1_qs_dict_g = jt1_qs_dict_g  # NEW
            frame.jt2_qs_dict_g = jt2_qs_dict_g  # NEW

        return frame

    element_processor = f2_rho_and_j if calculate_currents else f2_rho

    # setup window iterator
    window_iterator = WindowIterator(
        traj,
        width=N_tc,
        window_step=window_step,
        element_processor=element_processor
    )

    # define all pairs
    pairs = list(combinations_with_replacement(traj.atom_types, r=2))
    particle_counts = {key: len(val) for key, val in traj.atomic_indices.items()}

    # setup all time averager instances
    F_q_t_averager = {pair: TimeAverager(N_tc, n_qpoints) for pair in pairs}
    if calculate_currents:
        Cl_q_t_averager  = {pair: TimeAverager(N_tc, n_qpoints) for pair in pairs}
        Ct_q_t_averager  = {pair: TimeAverager(N_tc, n_qpoints) for pair in pairs}   # keep original Ct
        Ct1_q_t_averager = {pair: TimeAverager(N_tc, n_qpoints) for pair in pairs}   # NEW
        Ct2_q_t_averager = {pair: TimeAverager(N_tc, n_qpoints) for pair in pairs}   # NEW
    if calculate_incoherent:
        F_s_q_t_averager = {atom_type: TimeAverager(N_tc, n_qpoints) for atom_type in traj.atom_types}

    # define correlation function
    def calc_corr(window, time_i):
        f0 = window[0]
        fi = window[time_i]

        # density correlations (unchanged)
        for s1, s2 in pairs:
            Fqt = np.real(f0.rho_qs_dict[s1] * fi.rho_qs_dict[s2].conjugate())
            if s1 != s2:
                Fqt += np.real(f0.rho_qs_dict[s2] * fi.rho_qs_dict[s1].conjugate())
            F_q_t_averager[(s1, s2)].add_sample(time_i, Fqt)

        # current correlations
        if calculate_currents:
            for s1, s2 in pairs:
                if not currents_4group:
                    Clqt = np.real(f0.jz_qs_dict[s1] * fi.jz_qs_dict[s2].conjugate())

                    # original Ct (vector transverse dot)
                    Ctqt = 0.5 * np.real(np.sum(f0.jper_qs_dict[s1] *
                                                fi.jper_qs_dict[s2].conjugate(), axis=1))

                    # NEW: split transverse
                    Ct1qt = np.real(f0.jt1_qs_dict[s1] * fi.jt1_qs_dict[s2].conjugate())
                    Ct2qt = np.real(f0.jt2_qs_dict[s1] * fi.jt2_qs_dict[s2].conjugate())

                    if s1 != s2:
                        Clqt += np.real(f0.jz_qs_dict[s2] * fi.jz_qs_dict[s1].conjugate())
                        Ctqt += 0.5 * np.real(np.sum(f0.jper_qs_dict[s2] *
                                                     fi.jper_qs_dict[s1].conjugate(), axis=1))
                        Ct1qt += np.real(f0.jt1_qs_dict[s2] * fi.jt1_qs_dict[s1].conjugate())
                        Ct2qt += np.real(f0.jt2_qs_dict[s2] * fi.jt2_qs_dict[s1].conjugate())

                    Cl_q_t_averager[(s1, s2)].add_sample(time_i, Clqt)
                    Ct_q_t_averager[(s1, s2)].add_sample(time_i, Ctqt)
                    Ct1_q_t_averager[(s1, s2)].add_sample(time_i, Ct1qt)  # NEW
                    Ct2_q_t_averager[(s1, s2)].add_sample(time_i, Ct2qt)  # NEW

                else:
                    Clqt_sum  = np.zeros((n_qpoints,), dtype=np.float64)
                    Ctqt_sum  = np.zeros((n_qpoints,), dtype=np.float64)
                    Ct1qt_sum = np.zeros((n_qpoints,), dtype=np.float64)  # NEW
                    Ct2qt_sum = np.zeros((n_qpoints,), dtype=np.float64)  # NEW

                    for g in range(group_count):
                        jz0_s1 = f0.jz_qs_dict_g[g][s1]
                        jzi_s2 = fi.jz_qs_dict_g[g][s2]
                        jp0_s1 = f0.jper_qs_dict_g[g][s1]
                        jpi_s2 = fi.jper_qs_dict_g[g][s2]
                        jt10_s1 = f0.jt1_qs_dict_g[g][s1]   # NEW
                        jt1i_s2 = fi.jt1_qs_dict_g[g][s2]   # NEW
                        jt20_s1 = f0.jt2_qs_dict_g[g][s1]   # NEW
                        jt2i_s2 = fi.jt2_qs_dict_g[g][s2]   # NEW

                        Clqt_g  = np.real(jz0_s1 * jzi_s2.conjugate())
                        Ctqt_g  = 0.5 * np.real(np.sum(jp0_s1 * jpi_s2.conjugate(), axis=1))
                        Ct1qt_g = np.real(jt10_s1 * jt1i_s2.conjugate())     # NEW
                        Ct2qt_g = np.real(jt20_s1 * jt2i_s2.conjugate())     # NEW

                        if s1 != s2:
                            jz0_s2 = f0.jz_qs_dict_g[g][s2]
                            jzi_s1 = fi.jz_qs_dict_g[g][s1]
                            jp0_s2 = f0.jper_qs_dict_g[g][s2]
                            jpi_s1 = fi.jper_qs_dict_g[g][s1]
                            jt10_s2 = f0.jt1_qs_dict_g[g][s2]  # NEW
                            jt1i_s1 = fi.jt1_qs_dict_g[g][s1]  # NEW
                            jt20_s2 = f0.jt2_qs_dict_g[g][s2]  # NEW
                            jt2i_s1 = fi.jt2_qs_dict_g[g][s1]  # NEW

                            Clqt_g  += np.real(jz0_s2 * jzi_s1.conjugate())
                            Ctqt_g  += 0.5 * np.real(np.sum(jp0_s2 * jpi_s1.conjugate(), axis=1))
                            Ct1qt_g += np.real(jt10_s2 * jt1i_s1.conjugate())  # NEW
                            Ct2qt_g += np.real(jt20_s2 * jt2i_s1.conjugate())  # NEW

                        Clqt_sum  += Clqt_g
                        Ctqt_sum  += Ctqt_g
                        Ct1qt_sum += Ct1qt_g
                        Ct2qt_sum += Ct2qt_g

                    Cl_q_t_averager[(s1, s2)].add_sample(time_i, Clqt_sum)
                    Ct_q_t_averager[(s1, s2)].add_sample(time_i, Ctqt_sum)
                    Ct1_q_t_averager[(s1, s2)].add_sample(time_i, Ct1qt_sum)  # NEW
                    Ct2_q_t_averager[(s1, s2)].add_sample(time_i, Ct2qt_sum)  # NEW

        # incoherent part (unchanged)
        if calculate_incoherent:
            for atom_type in traj.atom_types:
                xi = fi.positions_by_type[atom_type]
                x0 = f0.positions_by_type[atom_type]
                Fsqt = np.real(calc_rho_q(xi - x0, q_points))
                F_s_q_t_averager[atom_type].add_sample(time_i, Fsqt)

    # run calculation (unchanged)
    logging_interval = 1000
    with concurrent.futures.ThreadPoolExecutor() as tpe:
        for window in window_iterator:
            if window[0].frame_index % logging_interval == 0:
                logger.info(f'Processing window {window[0].frame_index} to {window[-1].frame_index}')
            for _ in tpe.map(partial(calc_corr, window), range(len(window))):
                pass

    # collect results
    data_dict_corr = dict()
    time = delta_t * np.arange(N_tc, dtype=float)
    data_dict_corr['q_points'] = q_points
    data_dict_corr['time'] = time

    # --- density spectra (unchanged) ---
    F_q_t_tot = np.zeros((n_qpoints, N_tc))
    S_q_w_tot = np.zeros((n_qpoints, N_tc))
    for pair in pairs:
        key = '_'.join(pair)
        F_q_t = 1 / traj.n_atoms * F_q_t_averager[pair].get_average_all()
        w, S_q_w = fourier_cos_filon(F_q_t, delta_t)
        S_q_w = np.array(S_q_w)
        data_dict_corr['omega'] = w
        data_dict_corr[f'Fqt_coh_{key}'] = F_q_t
        data_dict_corr[f'Sqw_coh_{key}'] = S_q_w
        F_q_t_tot += F_q_t
        S_q_w_tot += S_q_w
    data_dict_corr['Fqt_coh'] = F_q_t_tot
    data_dict_corr['Sqw_coh'] = S_q_w_tot

    # --- current spectra (Ct1/Ct2 added) ---
    if calculate_currents:
        Cl_q_t_tot  = np.zeros((n_qpoints, N_tc))
        Ct_q_t_tot  = np.zeros((n_qpoints, N_tc))
        Ct1_q_t_tot = np.zeros((n_qpoints, N_tc))  # NEW
        Ct2_q_t_tot = np.zeros((n_qpoints, N_tc))  # NEW

        Cl_q_w_tot  = np.zeros((n_qpoints, N_tc))
        Ct_q_w_tot  = np.zeros((n_qpoints, N_tc))
        Ct1_q_w_tot = np.zeros((n_qpoints, N_tc))  # NEW
        Ct2_q_w_tot = np.zeros((n_qpoints, N_tc))  # NEW

        for pair in pairs:
            key = '_'.join(pair)

            Cl_q_t  = 1 / traj.n_atoms * Cl_q_t_averager[pair].get_average_all()
            Ct_q_t  = 1 / traj.n_atoms * Ct_q_t_averager[pair].get_average_all()
            Ct1_q_t = 1 / traj.n_atoms * Ct1_q_t_averager[pair].get_average_all()  # NEW
            Ct2_q_t = 1 / traj.n_atoms * Ct2_q_t_averager[pair].get_average_all()  # NEW

            _, Cl_q_w  = fourier_cos_filon(Cl_q_t,  delta_t)
            _, Ct_q_w  = fourier_cos_filon(Ct_q_t,  delta_t)
            _, Ct1_q_w = fourier_cos_filon(Ct1_q_t, delta_t)  # NEW
            _, Ct2_q_w = fourier_cos_filon(Ct2_q_t, delta_t)  # NEW

            data_dict_corr[f'Clqt_{key}']  = Cl_q_t
            data_dict_corr[f'Ctqt_{key}']  = Ct_q_t
            data_dict_corr[f'Ct1qt_{key}'] = Ct1_q_t  # NEW
            data_dict_corr[f'Ct2qt_{key}'] = Ct2_q_t  # NEW

            data_dict_corr[f'Clqw_{key}']  = Cl_q_w
            data_dict_corr[f'Ctqw_{key}']  = Ct_q_w
            data_dict_corr[f'Ct1qw_{key}'] = Ct1_q_w  # NEW
            data_dict_corr[f'Ct2qw_{key}'] = Ct2_q_w  # NEW

            Cl_q_t_tot  += Cl_q_t
            Ct_q_t_tot  += Ct_q_t
            Ct1_q_t_tot += Ct1_q_t
            Ct2_q_t_tot += Ct2_q_t

            Cl_q_w_tot  += Cl_q_w
            Ct_q_w_tot  += Ct_q_w
            Ct1_q_w_tot += Ct1_q_w
            Ct2_q_w_tot += Ct2_q_w

        data_dict_corr['Clqt']  = Cl_q_t_tot
        data_dict_corr['Ctqt']  = Ct_q_t_tot
        data_dict_corr['Ct1qt'] = Ct1_q_t_tot   # NEW
        data_dict_corr['Ct2qt'] = Ct2_q_t_tot   # NEW

        data_dict_corr['Clqw']  = Cl_q_w_tot
        data_dict_corr['Ctqw']  = Ct_q_w_tot
        data_dict_corr['Ct1qw'] = Ct1_q_w_tot   # NEW
        data_dict_corr['Ct2qw'] = Ct2_q_w_tot   # NEW

    # --- incoherent spectra (unchanged) ---
    if calculate_incoherent:
        Fs_q_t_tot = np.zeros((n_qpoints, N_tc))
        Ss_q_w_tot = np.zeros((n_qpoints, N_tc))
        for atom_type in traj.atom_types:
            Fs_q_t = 1 / traj.n_atoms * F_s_q_t_averager[atom_type].get_average_all()
            _, Ss_q_w = fourier_cos_filon(Fs_q_t, delta_t)
            data_dict_corr[f'Fqt_incoh_{atom_type}'] = Fs_q_t
            data_dict_corr[f'Sqw_incoh_{atom_type}'] = Ss_q_w
            Fs_q_t_tot += Fs_q_t
            Ss_q_w_tot += Ss_q_w

        data_dict_corr['Fqt_incoh'] = Fs_q_t_tot
        data_dict_corr['Sqw_incoh'] = Ss_q_w_tot
        data_dict_corr['Fqt'] = data_dict_corr['Fqt_coh'] + data_dict_corr['Fqt_incoh']
        data_dict_corr['Sqw'] = data_dict_corr['Sqw_coh'] + data_dict_corr['Sqw_incoh']
    else:
        data_dict_corr['Fqt'] = data_dict_corr['Fqt_coh'].copy()
        data_dict_corr['Sqw'] = data_dict_corr['Sqw_coh'].copy()

    # finalize
    result = DynamicSample(
        data_dict_corr,
        atom_types=traj.atom_types,
        pairs=pairs,
        particle_counts=particle_counts,
        cell=traj.cell,
        time_between_frames=delta_t,
        maximum_time_lag=delta_t * window_size,
        angular_frequency_resolution=dw,
        maximum_angular_frequency=w_max,
        number_of_frames=traj.number_of_frames_read
    )
    return result


def compute_static_structure_factors(
    traj: Trajectory,
    q_points: NDArray[float],
) -> StaticSample:
    r"""Compute static structure factors.  The results are returned in the
    form of a :class:`StaticSample <dynasor.sample.StaticSample>`
    object.

    Parameters
    ----------
    traj
        Input trajectory
    q_points
        Array of q-points in units of rad/Å with shape ``(N_qpoints, 3)`` in Cartesian coordinates
    """
    # sanity check input args
    if q_points.shape[1] != 3:
        raise ValueError('q-points array has the wrong shape.')

    n_qpoints = q_points.shape[0]
    logger.info(f'Number of q-points: {n_qpoints}')

    # define all pairs
    pairs = list(combinations_with_replacement(traj.atom_types, r=2))
    particle_counts = {key: len(val) for key, val in traj.atomic_indices.items()}
    logger.debug('Considering pairs:')
    for pair in pairs:
        logger.debug(f'  {pair}')

    # processing function
    def f2_rho(frame):
        rho_qs_dict = dict()
        for atom_type in frame.positions_by_type.keys():
            x = frame.positions_by_type[atom_type]
            rho_qs_dict[atom_type] = calc_rho_q(x, q_points)
        frame.rho_qs_dict = rho_qs_dict
        return frame

    # setup averager
    Sq_averager = dict()
    for pair in pairs:
        Sq_averager[pair] = TimeAverager(1, n_qpoints)  # time average with only timelag=0

    # main loop
    for frame in traj:
        f2_rho(frame)
        logger.debug(f'Processing frame {frame.frame_index}')

        for s1, s2 in pairs:
            Sq_pair = np.real(frame.rho_qs_dict[s1] * frame.rho_qs_dict[s2].conjugate())
            if s1 != s2:
                Sq_pair += np.real(frame.rho_qs_dict[s2] * frame.rho_qs_dict[s1].conjugate())
            Sq_averager[(s1, s2)].add_sample(0, Sq_pair)

    # collect results
    data_dict = dict()
    data_dict['q_points'] = q_points
    S_q_tot = np.zeros((n_qpoints, 1))
    for s1, s2 in pairs:
        Sq = 1 / traj.n_atoms * Sq_averager[(s1, s2)].get_average_at_timelag(0).reshape(-1, 1)
        data_dict[f'Sq_{s1}_{s2}'] = Sq
        S_q_tot += Sq
    data_dict['Sq'] = S_q_tot

    # finalize results
    result = StaticSample(
        data_dict,
        atom_types=traj.atom_types,
        pairs=pairs,
        particle_counts=particle_counts,
        cell=traj.cell,
        number_of_frames=traj.number_of_frames_read
    )
    return result


def compute_spectral_energy_density(
    traj: Trajectory,
    ideal_supercell: Atoms,
    primitive_cell: Atoms,
    q_points: NDArray[float],
    dt: float,
    partial: bool = False
) -> Tuple[NDArray[float], NDArray[float]]:
    r"""
    Compute the spectral energy density (SED) at specific q-points. The results
    are returned in the form of a tuple, which comprises the angular
    frequencies in an array of length ``N_times`` in units of rad/fs and the
    SED in units of eV/(rad/fs) as an array of shape ``(N_qpoints, N_times)``.
    The normalization is chosen such that integrating the SED of a q-point
    together with the supplied angular frequenceies omega (rad/fs) yields
    1/2kBT * number of bands (where number of bands = len(prim) * 3)

    More details can be found in Thomas *et al.*, Physical Review B **81**, 081411 (2010),
    which should be cited when using this function along with the dynasor reference.

    **Note 1:**
    SED analysis is only suitable for crystalline materials without diffusion as
    atoms are assumed to move around fixed reference positions throughout the entire trajectory.

    **Note 2:**
    This implementation reads the full trajectory and can thus consume a lot of memory.

    Parameters
    ----------
    traj
        Input trajectory
    ideal_supercell
        Ideal structure defining the reference positions. Do not change the
        masses in the ASE atoms objects to dynasor internal units, this will be
        done internally
    primitive_cell
        Underlying primitive structure. Must be aligned correctly with :attr:`ideal_supercell`.
    q_points
        Array of q-points in units of rad/Å with shape ``(N_qpoints, 3)`` in Cartesian coordinates
    dt
        Time difference in femtoseconds between two consecutive snapshots in
        the trajectory. Note that you should not change :attr:`dt` if you change
        :attr:`frame_step <dynasor.trajectory.Trajectory.frame_step>` in :attr:`traj`.
    partial
        If True the SED will be returned decomposed per basis and Cartesian direction.
        The shape is ``(N_qpoints, N_frequencies, len(primitive_cell), 3)``
    """

    delta_t = traj.frame_step * dt

    # logger
    logger.info('Running SED')
    logger.info(f'Time between consecutive frames (dt * frame_step): {delta_t} fs')
    logger.info(f'Number of atoms in primitive_cell: {len(primitive_cell)}')
    logger.info(f'Number of atoms in ideal_supercell: {len(ideal_supercell)}')
    logger.info(f'Number of q-points: {q_points.shape[0]}')

    # check that the ideal supercell agrees with traj
    if traj.n_atoms != len(ideal_supercell):
        raise ValueError('ideal_supercell must contain the same number of atoms as the trajectory.')

    if len(primitive_cell) >= len(ideal_supercell):
        raise ValueError('primitive_cell contains more atoms than ideal_supercell.')

    # colllect all velocities, and scale with sqrt(masses)
    masses = ideal_supercell.get_masses().reshape(-1, 1) / fs**2  # From Dalton to dmu
    velocities = []
    for it, frame in enumerate(traj):
        logger.debug(f'Reading frame {it}')
        if frame.velocities_by_type is None:
            raise ValueError(f'Could not read velocities from frame {it}')
        v = frame.get_velocities_as_array(traj.atomic_indices)  # in Å/fs
        velocities.append(np.sqrt(masses) * v)
    logger.info(f'Number of snapshots: {len(velocities)}')

    # Perform the FFT on the last axis for extra speed (maybe not needed)
    N_samples = len(velocities)
    velocities = np.array(velocities)
    velocities = velocities.transpose(1, 2, 0).copy()
    velocities = np.fft.rfft(velocities, axis=2)

    # Calcualte indices and offsets needed for the sed method
    indices, offsets = get_index_offset(ideal_supercell, primitive_cell)

    # Phase factor for use in FT. #qpoints x #atoms in supercell
    cell_positions = np.dot(offsets, primitive_cell.cell)
    phase = np.dot(q_points, cell_positions.T)  # #qpoints x #unit cells
    phase_factors = np.exp(1.0j * phase)

    # Map offsets -> index
    offset_dict = {off: n for n, off in enumerate(set(tuple(offset) for offset in offsets))}

    # Shapes
    n_super, _, n_w = velocities.shape
    n_qpts = len(q_points)
    n_prim = len(primitive_cell)
    n_offsets = len(offset_dict)

    new_velocities = np.zeros((n_w, 3, n_prim, n_offsets), dtype=velocities.dtype)

    for i in range(n_super):
        j = indices[i]
        n = offset_dict[tuple(offsets[i])]
        new_velocities[:, :, j, n] = velocities[i].T

    velocities = new_velocities

    new_phase_factors = np.zeros((n_qpts, n_prim, n_offsets), dtype=phase_factors.dtype)

    for i in range(n_super):
        j = indices[i]
        n = offset_dict[tuple(offsets[i])]
        new_phase_factors[:, j, n] = phase_factors[:, i]

    phase_factors = new_phase_factors

    density = _sed_inner_loop(phase_factors, velocities)

    if not partial:
        density = np.sum(density, axis=(2, 3))

    density = density * delta_t**2
    density = density / (N_samples * delta_t)
    density = density / (n_super / n_prim)
    density = density / (2 * np.pi)

    w = 2 * np.pi * np.fft.rfftfreq(N_samples, delta_t)  # rad/fs

    return w, density


@numba.njit(parallel=True, fastmath=True)
def _sed_inner_loop(phase_factors, velocities):
    """This numba function calculates the spatial FT using precomputed phase factors"""
    n_qpts = phase_factors.shape[0]
    n_prim = phase_factors.shape[1]
    n_super = phase_factors.shape[2]

    n_freqs = velocities.shape[0]

    density = np.zeros((n_qpts, n_freqs, n_prim, 3), dtype=np.float64)

    for w in numba.prange(n_freqs):
        for k in range(n_qpts):
            for a in range(3):
                for b in range(n_prim):
                    tmp = 0.0j
                    for n in range(n_super):
                        tmp += phase_factors[k, b, n] * velocities[w, a, b, n]
                    density[k, w, b, a] += np.abs(tmp) ** 2
    return density