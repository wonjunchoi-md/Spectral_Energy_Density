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


def compute_dynamic_structure_factors(
    traj: Trajectory,
    q_points: NDArray[float],
    dt: float,
    window_size: int,
    window_step: int = 1,
    calculate_currents: bool = False,
    calculate_incoherent: bool = False,
    # --- MATLAB-style option ---
    currents_4group: bool = False,  # current correlation을 4-group으로 나눠 계산 후 그룹합(교차항 제외)
) -> DynamicSample:
    """Compute dynamic structure factors.  The results are returned in the
    form of a :class:`DynamicSample <dynasor.sample.DynamicSample>`
    object.

    Parameters
    ----------
    traj
        Input trajectory
    q_points
        Array of q-points in units of rad/Å with shape ``(N_qpoints, 3)`` in Cartesian coordinates
    dt
        Time difference in femtoseconds between two consecutive snapshots
        in the trajectory. Note that you should *not* change :attr:`dt` if you change
        :attr:`frame_step <dynasor.trajectory.Trajectory.frame_step>` in :attr:`traj`.
    window_size
        Length of the trajectory frame window to use for time correlation calculation.
        It is expressed in terms of the number of time lags to consider
        and thus determines the smallest frequency resolved.
    window_step
        Window step (or stride) given as the number of frames between consecutive trajectory
        windows. This parameter does *not* affect the time between consecutive frames in the
        calculation. If, e.g., :attr:`window_step` > :attr:`window_size`, some frames will not
        be used.
    calculate_currents
        Calculate the current correlations. Requires velocities to be available in :attr:`traj`.
    calculate_incoherent
        Calculate the incoherent part (self-part) of :math:`F_incoh`.
    currents_4group
        If True (and calculate_currents=True), current correlations are computed
        in 4 groups (by global atom index mod 4) and summed across groups,
        mimicking your MATLAB "4-group" workflow (no cross terms between groups).
        Density correlations F(q,t) are still computed on the full set (unchanged).
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
        if currents_4group:
            logger.info('  Using MATLAB-style 4-group partition for current correlations (sum over groups; no cross terms)')
    if calculate_incoherent:
        logger.info('Calculating incoherent part (self-part) of correlations')

    # log some info regarding q-points
    logger.info(f'Number of q-points: {n_qpoints}')

    q_directions = q_points.copy()
    q_distances = np.linalg.norm(q_points, axis=1)
    nonzero = q_distances > 0
    q_directions[nonzero] /= q_distances[nonzero].reshape(-1, 1)

    # --- MATLAB-style 4-group partition: per atom_type, choose atoms by (global_index % 4) ---
    group_count = 4
    group_local_idx = None
    if currents_4group:
        group_local_idx = [dict() for _ in range(group_count)]
        for atom_type in traj.atom_types:
            gidx = np.asarray(traj.atomic_indices[atom_type], dtype=np.int64)  # global indices (traj ordering)
            local = np.arange(len(gidx), dtype=np.int64)  # local indices in positions_by_type[atom_type]
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

        # 4-group stores: list[group] -> dict[atom_type] -> array
        if currents_4group:
            jz_qs_dict_g = [dict() for _ in range(group_count)]
            jper_qs_dict_g = [dict() for _ in range(group_count)]

        for atom_type in frame.positions_by_type.keys():
            x_all = frame.positions_by_type[atom_type]
            v_all = frame.velocities_by_type[atom_type]

            # Density correlations: keep original "full set" behavior (unchanged)
            rho_qs_dict[atom_type] = calc_rho_q(x_all, q_points)

            if not currents_4group:
                rho_qs, j_qs = calc_rho_j_q(x_all, v_all, q_points)
                jz_qs = np.sum(j_qs * q_directions, axis=1)
                jper_qs = j_qs - (jz_qs[:, None] * q_directions)

                # NOTE: rho_qs is same as calc_rho_q(x_all, q_points), but keep as original pattern
                # (we already computed rho_qs_dict above)
                jz_qs_dict[atom_type] = jz_qs
                jper_qs_dict[atom_type] = jper_qs
            else:
                # MATLAB-style: compute j(q,t) separately per group and later sum correlations across groups
                for g in range(group_count):
                    sel = group_local_idx[g][atom_type]
                    if sel.size == 0:
                        jz_qs_dict_g[g][atom_type] = np.zeros((n_qpoints,), dtype=np.complex128)
                        jper_qs_dict_g[g][atom_type] = np.zeros((n_qpoints, 3), dtype=np.complex128)
                        continue

                    x = x_all[sel]
                    v = v_all[sel]
                    _, j_qs = calc_rho_j_q(x, v, q_points)  # (#q, 3)

                    jz_qs = np.sum(j_qs * q_directions, axis=1)
                    jper_qs = j_qs - (jz_qs[:, None] * q_directions)

                    jz_qs_dict_g[g][atom_type] = jz_qs
                    jper_qs_dict_g[g][atom_type] = jper_qs

        frame.rho_qs_dict = rho_qs_dict

        if not currents_4group:
            frame.jz_qs_dict = jz_qs_dict
            frame.jper_qs_dict = jper_qs_dict
        else:
            frame.jz_qs_dict_g = jz_qs_dict_g
            frame.jper_qs_dict_g = jper_qs_dict_g

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
    logger.debug('Considering pairs:')
    for pair in pairs:
        logger.debug(f'  {pair}')

    # setup all time averager instances
    F_q_t_averager = {pair: TimeAverager(N_tc, n_qpoints) for pair in pairs}
    if calculate_currents:
        Cl_q_t_averager = {pair: TimeAverager(N_tc, n_qpoints) for pair in pairs}
        Ct_q_t_averager = {pair: TimeAverager(N_tc, n_qpoints) for pair in pairs}
    if calculate_incoherent:
        F_s_q_t_averager = {atom_type: TimeAverager(N_tc, n_qpoints) for atom_type in traj.atom_types}

    # define correlation function
    def calc_corr(window, time_i):
        # Calculate correlations between two frames in the window without normalization 1/N
        f0 = window[0]
        fi = window[time_i]

        # --- density correlations (unchanged) ---
        for s1, s2 in pairs:
            Fqt = np.real(f0.rho_qs_dict[s1] * fi.rho_qs_dict[s2].conjugate())
            if s1 != s2:
                Fqt += np.real(f0.rho_qs_dict[s2] * fi.rho_qs_dict[s1].conjugate())
            F_q_t_averager[(s1, s2)].add_sample(time_i, Fqt)

        # --- current correlations ---
        if calculate_currents:
            for s1, s2 in pairs:
                if not currents_4group:
                    Clqt = np.real(f0.jz_qs_dict[s1] * fi.jz_qs_dict[s2].conjugate())
                    Ctqt = 0.5 * np.real(np.sum(f0.jper_qs_dict[s1] *
                                                fi.jper_qs_dict[s2].conjugate(), axis=1))
                    if s1 != s2:
                        Clqt += np.real(f0.jz_qs_dict[s2] * fi.jz_qs_dict[s1].conjugate())
                        Ctqt += 0.5 * np.real(np.sum(f0.jper_qs_dict[s2] *
                                                     fi.jper_qs_dict[s1].conjugate(), axis=1))

                    Cl_q_t_averager[(s1, s2)].add_sample(time_i, Clqt)
                    Ct_q_t_averager[(s1, s2)].add_sample(time_i, Ctqt)

                else:
                    # MATLAB-like: sum_g < j_g(0) * j_g(t)* >, i.e. no cross-group terms
                    Clqt_sum = np.zeros((n_qpoints,), dtype=np.float64)
                    Ctqt_sum = np.zeros((n_qpoints,), dtype=np.float64)

                    for g in range(group_count):
                        jz0_s1 = f0.jz_qs_dict_g[g][s1]
                        jzi_s2 = fi.jz_qs_dict_g[g][s2]
                        jp0_s1 = f0.jper_qs_dict_g[g][s1]
                        jpi_s2 = fi.jper_qs_dict_g[g][s2]

                        Clqt_g = np.real(jz0_s1 * jzi_s2.conjugate())
                        Ctqt_g = 0.5 * np.real(np.sum(jp0_s1 * jpi_s2.conjugate(), axis=1))

                        if s1 != s2:
                            jz0_s2 = f0.jz_qs_dict_g[g][s2]
                            jzi_s1 = fi.jz_qs_dict_g[g][s1]
                            jp0_s2 = f0.jper_qs_dict_g[g][s2]
                            jpi_s1 = fi.jper_qs_dict_g[g][s1]

                            Clqt_g += np.real(jz0_s2 * jzi_s1.conjugate())
                            Ctqt_g += 0.5 * np.real(np.sum(jp0_s2 * jpi_s1.conjugate(), axis=1))

                        Clqt_sum += Clqt_g
                        Ctqt_sum += Ctqt_g

                    Cl_q_t_averager[(s1, s2)].add_sample(time_i, Clqt_sum)
                    Ct_q_t_averager[(s1, s2)].add_sample(time_i, Ctqt_sum)

        # --- incoherent part (unchanged) ---
        if calculate_incoherent:
            for atom_type in traj.atom_types:
                xi = fi.positions_by_type[atom_type]
                x0 = f0.positions_by_type[atom_type]
                Fsqt = np.real(calc_rho_q(xi - x0, q_points))
                F_s_q_t_averager[atom_type].add_sample(time_i, Fsqt)

    # run calculation
    logging_interval = 1000
    with concurrent.futures.ThreadPoolExecutor() as tpe:
        for window in window_iterator:
            logger.debug(f'Processing window {window[0].frame_index} to {window[-1].frame_index}')

            if window[0].frame_index % logging_interval == 0:
                logger.info(f'Processing window {window[0].frame_index} to {window[-1].frame_index}')

            for _ in tpe.map(partial(calc_corr, window), range(len(window))):
                pass

    # collect results into dict with numpy arrays (n_qpoints, N_tc)
    data_dict_corr = dict()
    time = delta_t * np.arange(N_tc, dtype=float)
    data_dict_corr['q_points'] = q_points
    data_dict_corr['time'] = time

    # --- density spectra ---
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

    # --- current spectra ---
    if calculate_currents:
        Cl_q_t_tot = np.zeros((n_qpoints, N_tc))
        Ct_q_t_tot = np.zeros((n_qpoints, N_tc))
        Cl_q_w_tot = np.zeros((n_qpoints, N_tc))
        Ct_q_w_tot = np.zeros((n_qpoints, N_tc))

        for pair in pairs:
            key = '_'.join(pair)
            Cl_q_t = 1 / traj.n_atoms * Cl_q_t_averager[pair].get_average_all()
            Ct_q_t = 1 / traj.n_atoms * Ct_q_t_averager[pair].get_average_all()
            _, Cl_q_w = fourier_cos_filon(Cl_q_t, delta_t)
            _, Ct_q_w = fourier_cos_filon(Ct_q_t, delta_t)

            data_dict_corr[f'Clqt_{key}'] = Cl_q_t
            data_dict_corr[f'Ctqt_{key}'] = Ct_q_t
            data_dict_corr[f'Clqw_{key}'] = Cl_q_w
            data_dict_corr[f'Ctqw_{key}'] = Ct_q_w

            Cl_q_t_tot += Cl_q_t
            Ct_q_t_tot += Ct_q_t
            Cl_q_w_tot += Cl_q_w
            Ct_q_w_tot += Ct_q_w

        data_dict_corr['Clqt'] = Cl_q_t_tot
        data_dict_corr['Ctqt'] = Ct_q_t_tot
        data_dict_corr['Clqw'] = Cl_q_w_tot
        data_dict_corr['Ctqw'] = Ct_q_w_tot

    # --- incoherent spectra ---
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

    # finalize results with additional meta data
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