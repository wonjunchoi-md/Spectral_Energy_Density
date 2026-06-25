#!/usr/bin/python3

import sys
import argparse
import numpy as np

from dynasor.logging_tools import logger, set_logging_level
from dynasor.qpoints import get_spherical_qpoints
from dynasor.trajectory import Trajectory
from dynasor.correlation_functions import compute_dynamic_structure_factors
from dynasor.post_processing import get_spherically_averaged_sample_binned


def main():

    parser = argparse.ArgumentParser(
        description='dynasor is a simple tool for calculating total and partial dynamic structure'
        ' factors as well as current correlation functions from molecular dynamics simulations.'
        ' The main input consists of a trajectory output from a MD simulation, i.e., a file'
        ' containing snapshots of the particle coordinates and optionally velocities that'
        ' correspond to consecutive, equally spaced points in (simulation) time.'
        '\n'
        ' Dynasor has recently been updated. Some of the old options have new names and'
        ' some new options have been added. If your script has stopped working, check the'
        ' options with "dynasor --help". In addition, we would like to let you know that'
        ' dynasor now has a Python interface, which gives you acces to more'
        ' functionality and options.')

    iogroup = parser.add_argument_group(
        'Input/output options',
        'Options controlling input and output, files and fileformats.')
    iogroup.add_argument(
        '-f', '--trajectory', type=str, metavar='TRAJECTORY_FILE',
        help='Molecular dynamics trajectory file to be analyzed.'
        ' Supported formats depends on MDAnalysis. As a fallback, a lammps-trajectory parser'
        ' implemented in Python is also available as well as an extended-xyz reader based on ASE')
    iogroup.add_argument(
        '--trajectory-format', type=str, metavar='TRAJECTORY_FORMAT',
        help='Format of trajectory. Choose from: "lammps_internal", "extxyz", or one'
        ' of the formats supported by MDAnalysis (except "lammpsdump", which is'
        ' called through "lammps_mdanalysis" to avoid ambiguity)')
    iogroup.add_argument(
        '--length-unit', type=str, metavar='LENGTH_UNIT', default='Angstrom',
        help='Length unit of trajectory ("Angstrom", "nm", "pm", "fm"). Necessary for correct '
        'conversion to internal dynasor units if the trajectory file does not contain '
        'unit information.')
    iogroup.add_argument(
        '--time-unit', type=str, metavar='TIME_UNIT', default='fs',
        help='Time unit of trajectory ("fs", "ps", "ns"). Necessary for correct conversion to '
        'internal dynasor units if the trajectory file does not contain unit information.')
    iogroup.add_argument(
        '-n', '--index', type=str, metavar='INDEX_FILE',
        help='Optional index file (think Gromacs NDX-style) for specifying atom types. Atoms are'
        ' indexed from 1 up to N (total number of atoms). It is possible to index only a subset of'
        ' all atoms, and atoms can be indexed in more than one group. If no index file is provided,'
        ' all atoms will be considered identical.')
    iogroup.add_argument(
        '--outfile', type=str, metavar='FILE',
        help='Write output to FILE as a numpy npz file')

    qspace = parser.add_argument_group(
        'General q-space options',
        'Options controlling general aspects for how q-space should be sampled and collected.')
    qspace.add_argument(
        '--q-sampling', type=str,
        metavar='STYLE', default='isotropic',
        help='Possible values are "isotropic" (default) for sampling isotropic systems'
        ' (as liquids), and "line" to sample uniformly along a certain direction in q-space. ')
    defval = 80
    qspace.add_argument(
        '--q-bins',
        metavar='BINS', type=int, default=defval,
        help='Number of "radial" bins to use (between 0 and largest |q|-value) when collecting'
        f' resulting average. Default value is {defval}.')

    qiso = parser.add_argument_group('Isotropic q-space sampling')
    defval = 20000
    qiso.add_argument(
        '--max-q-points', metavar='QPOINTS', type=int,
        default=defval,
        help='Maximum number of points used to sample q-space. Puts an (approximate) upper'
        f' limit by randomly selecting. points. Default value is {defval}.')
    defval = 60
    qiso.add_argument(
        '--q-max', metavar='QMAX', type=int, default=defval,
        help='Largest q-value to consider in units of "2*pi*Å^-1".'
        ' Default value for QMAX is {defval}.')

    qline = parser.add_argument_group('Line-style q-space sampling')
    qline.add_argument(
        '--q-direction', metavar='QDIRECTION',
        help='Direction along which to sample. QPOINTS points will be evenly placed between'
        ' 0,0,0 and QDIRECTION. Given as three comma separated values.')
    defval = 100
    qline.add_argument(
        '--q-points', metavar='QPOINTS', type=int, default=defval,
        help=f'Number of q-points to sample along line. Default: {defval}')

    tgroup = parser.add_argument_group(
        'Time-related options',
        'Options controlling timestep, length and shape of trajectory frame window, etc.')
    tgroup.add_argument(
        '--time-window', metavar='TIME_WINDOW', type=int,
        help='The length of the trajectory frame window to use for time correlation calculation.'
        ' It is expressed in number of frames and determines, among other things, the smallest'
        ' frequency that can be resolved. If no TIME_WINDOW is provided, only static (t=0)'
        ' correlations will be calculated')
    defval = 100
    tgroup.add_argument(
        '--max-frames', metavar='FRAMES', type=int, default=defval,
        help='Limits the total number of trajectory frames read to FRAMES.'
        f' The default value is {defval}.')
    defval = 1
    tgroup.add_argument(
        '--step', metavar='STEP', type=int, default=defval,
        help='Only use every STEP-th trajectory frame. The default STEP is {defval}, meaning'
        ' every frame is processed. STEP affects dt and hence the smallest time resolved.')
    defval = 1
    tgroup.add_argument(
        '--stride', metavar='STRIDE', type=int, default=defval,
        help='STRIDE number of frames between consecutive trajectory windows. This does not affect'
        ' dt. If e.g. STRIDE > TIME_CORR_STEPS, some frames will be completely unused.')
    tgroup.add_argument(
        '--dt', metavar='DELTATIME', type=float,
        help='Explicitly sets the time difference between two consecutively processed'
        ' trajectory frames to DELTATIME (femtoseconds). ')

    options = parser.add_argument_group('General processing options')
    options.add_argument(
        '--calculate-incoherent',
        action='store_true', default=False,
        help='Calculate the incoherent part. Default is False.')
    options.add_argument(
        '--calculate-currents',
        action='store_true', default=False,
        help='Calculate the current (velocity) correlations,  Default is False.')

    parser.add_argument(
        '-q', '--quiet', action='count', default=0,
        help='Increase quietness (opposite of verbosity).')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Increase verbosity (opposite of quietness).')

    args = parser.parse_args()

    # set log level
    quietness = args.quiet - args.verbose
    if quietness < 0:
        log_level = 'DEBUG'
    elif quietness == 0:
        log_level = 'INFO'
    elif quietness == 1:
        log_level = 'WARN'
    elif quietness == 2:
        log_level = 'ERROR'
    else:
        log_level = 'CRITICAL'
    set_logging_level(log_level)

    # parse args
    if args.trajectory is None:
        logger.error('A trajectory must be specified. Use option -f')
        sys.exit(1)

    if not args.outfile:
        logger.error('An output file must be specified. Use option --outfile')
        sys.exit(1)

    if args.dt is None:
        logger.info('No value set for dt. Setting to 1 fs. Note that this is irrelevant when only '
                    'computing static structure factors, i.e., if time_window is not set.')
        args.dt = 1

    # setup Trajectory
    traj = Trajectory(args.trajectory,
                      trajectory_format=args.trajectory_format,
                      atomic_indices=args.index,
                      length_unit=args.length_unit,
                      time_unit=args.time_unit,
                      frame_stop=args.max_frames,
                      frame_step=args.step,
                      )

    # setup q-points
    if args.q_sampling == 'line':
        q_dir = np.array(args.q_direction)
        n_qpoints = args.q_points
        q_points = np.array([i * q_dir for i in np.linspace(0, 1, n_qpoints)])
    elif args.q_sampling == 'isotropic':
        q_points = get_spherical_qpoints(traj.cell, args.q_max, args.max_q_points)

    # run dynasor calculation
    sample = compute_dynamic_structure_factors(
        traj, q_points=q_points,
        dt=args.dt, window_size=args.time_window, window_step=args.stride,
        calculate_currents=args.calculate_currents,
        calculate_incoherent=args.calculate_incoherent,
        )

    # save results to file
    if args.q_sampling == 'isotropic':
        sample = get_spherically_averaged_sample_binned(sample, num_q_bins=args.q_bins)
    sample.write_to_npz(args.outfile)
