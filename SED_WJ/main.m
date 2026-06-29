%% ================================================================
%  SED Analysis — main.m
% ================================================================

%% ===== Config ===================================================
cfg.folderPath = '';
cfg.primFile   = 'prim_no_H.xyz';
cfg.timeStepFs = 40;       % dump 간격 [fs]  (dt=0.5fs × dump_every=80)
cfg.numSplits  = 1;
cfg.maxSteps   = 0;        % 읽을 최대 프레임 수 (0 = 전체)
% cfg.outputRoot = '';     % 비우면 SED_WJ/SED_outputs/ 자동 생성

%% ===== Pipeline =================================================

cfg            = prepare_output_dir(cfg);
[atoms, mData, folderPath] = read_trajectory(cfg.folderPath, cfg.maxSteps);
prim           = read_prim_xyz(fullfile(folderPath, cfg.primFile));
ref            = build_reference(prim, mData, atoms);
q_reduced      = (0 : floor(ref.N_UC/2)) / ref.N_UC;
q_cart         = make_q_path(ref, q_reduced);
mdata          = compute_SED(atoms, ref, q_cart, cfg);
mdata.q_reduced = q_reduced;
save_sed_result(mdata, cfg);
write_replot_script(cfg.outputDir);
plot_SED(mdata, cfg);



%% ================================================================
%  Local Functions
%% ================================================================

% ── 1. Trajectory reader ─────────────────────────────────────────
function [atoms, mData, folderPath] = read_trajectory(folderPath, maxSteps)
    if isempty(folderPath)
        folderPath = uigetdir(pwd, 'Select data folder');
        if folderPath == 0, folderPath = pwd; end
    end
    if nargin < 2, maxSteps = 0; end

    dumpFiles = dir(fullfile(folderPath, '*.lammpstrj'));
    if isempty(dumpFiles)
        error('No *.lammpstrj found in %s', folderPath);
    end
    dataFiles = dir(fullfile(folderPath, '*.data'));
    if isempty(dataFiles)
        dataFiles = dir(fullfile(folderPath, 'data.*'));
    end

    fprintf('Reading %s ...\n', dumpFiles(1).name);
    out            = read_combined_dump(fullfile(folderPath, dumpFiles(1).name), maxSteps);
    atoms.pos      = out.pos;
    atoms.vel      = out.vel;
    atoms.numAtoms = out.nAtoms;
    atoms.numSteps = out.nSteps;

    % 격자 상수는 항상 dump 박스에서 (NPT 평형 후 실제 MD 박스)
    mData = parse_box_from_dump(fullfile(folderPath, dumpFiles(1).name));

    % 질량은 .data 파일에서 (더 정확한 타입별 질량)
    if ~isempty(dataFiles)
        tmp = parse_data_file(fullfile(folderPath, dataFiles(1).name));
        mData.masses = tmp.masses;
    else
        fprintf('[Warning] No .data file — masses will come from prim.xyz\n');
    end
    fprintf('Loaded: %d atoms × %d steps\n', atoms.numAtoms, atoms.numSteps);
end


function out = read_combined_dump(filename, maxSteps)
% ITEM: ATOMS type id x y z vx vy vz
    if nargin < 2, maxSteps = 0; end
    fid = fopen(filename, 'r');
    if fid < 0, error('Cannot open %s', filename); end
    pos_cells = {}; vel_cells = {}; nAtoms = 0; step = 0;
    t0 = tic;
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if contains(line, 'NUMBER OF ATOMS')
            nAtoms = str2double(fgetl(fid));
        elseif contains(line, 'ITEM: ATOMS')
            step = step + 1;
            if maxSteps > 0 && step > maxSteps, break; end
            data = textscan(fid, '%f %f %f %f %f %f %f %f', nAtoms);
            pos_cells{step} = [data{1}, data{3}, data{4}, data{5}];
            vel_cells{step} = [data{1}, data{6}, data{7}, data{8}];
            if mod(step, 50) == 0
                fprintf('  frame %d  (%.0fs elapsed)\n', step, toc(t0));
            end
        end
    end
    fclose(fid);
    fprintf('  total %d frames read in %.0fs\n', step, toc(t0));
    nSteps = numel(pos_cells);
    pos = zeros(nAtoms, 4, nSteps);
    vel = zeros(nAtoms, 4, nSteps);
    for s = 1:nSteps
        pos(:,:,s) = pos_cells{s};
        vel(:,:,s) = vel_cells{s};
    end
    out.pos = pos; out.vel = vel; out.nAtoms = nAtoms; out.nSteps = nSteps;
end


function mData = parse_box_from_dump(filename)
    mData = struct('Lx',NaN,'Ly',NaN,'Lz',NaN,'masses',[],'chainDir','z');
    fid = fopen(filename, 'r');
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if contains(line, 'BOX BOUNDS')
            v = sscanf(fgetl(fid), '%f %f'); mData.Lx = v(2)-v(1);
            v = sscanf(fgetl(fid), '%f %f'); mData.Ly = v(2)-v(1);
            v = sscanf(fgetl(fid), '%f %f'); mData.Lz = v(2)-v(1);
            break;
        end
    end
    fclose(fid);
    dirs = 'xyz'; [~,d] = max([mData.Lx mData.Ly mData.Lz]);
    mData.chainDir = dirs(d);
    fprintf('[DumpBox] Lx=%.2f Ly=%.2f Lz=%.2f | chain=%s\n', ...
        mData.Lx, mData.Ly, mData.Lz, mData.chainDir);
end


function mData = parse_data_file(dataPath)
    fid = fopen(dataPath, 'r');
    C = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', ''); C = C{1};
    fclose(fid);

    mData = struct('Lx',NaN,'Ly',NaN,'Lz',NaN,'masses',[],'chainDir','z');

    for i = 1:numel(C)
        if contains(C{i},'xlo xhi'), v=sscanf(C{i},'%f %f'); mData.Lx=diff(v);
        elseif contains(C{i},'ylo yhi'), v=sscanf(C{i},'%f %f'); mData.Ly=diff(v);
        elseif contains(C{i},'zlo zhi'), v=sscanf(C{i},'%f %f'); mData.Lz=diff(v);
        end
    end

    massLine = find(strcmpi(strtrim(C), 'Masses'), 1);
    if ~isempty(massLine)
        i = massLine + 1;
        while i <= numel(C) && isempty(strtrim(C{i})), i = i+1; end
        while i <= numel(C)
            line = strtrim(regexprep(C{i}, '#.*', ''));
            if isempty(line), i = i+1; continue; end
            nums = str2double(strsplit(line)); nums = nums(~isnan(nums));
            if numel(nums) < 2, break; end
            mData.masses(end+1) = nums(2);
            i = i+1;
        end
    end

    dirs = 'xyz'; [~,d] = max([mData.Lx mData.Ly mData.Lz]);
    mData.chainDir = dirs(d);
    fprintf('[DataFile] Lx=%.2f Ly=%.2f Lz=%.2f | chain=%s | masses=%s\n', ...
        mData.Lx, mData.Ly, mData.Lz, mData.chainDir, mat2str(round(mData.masses)));
end


% ── 2. Primitive cell reader ──────────────────────────────────────
function prim = read_prim_xyz(xyzFile)
    fid = fopen(xyzFile, 'r');
    if fid < 0, error('Cannot open %s', xyzFile); end
    n_atoms = str2double(strtrim(fgetl(fid)));
    comment = fgetl(fid);
    fclose(fid);

    tok = regexp(comment, 'Lattice="([^"]+)"', 'tokens', 'once');
    if isempty(tok), error('Lattice= not found in %s', xyzFile); end
    cell_mat = reshape(str2double(strsplit(strtrim(tok{1}))), 3, 3)';

    prim.n_atoms   = n_atoms;
    prim.cell_diag = [cell_mat(1,1) cell_mat(2,2) cell_mat(3,3)];
    fprintf('[PrimCell] %d atoms | a=%.3f b=%.3f c=%.3f Å\n', ...
        n_atoms, prim.cell_diag(1), prim.cell_diag(2), prim.cell_diag(3));
end


% ── 3. Reference (ideal lattice positions R_n) ───────────────────
function ref = build_reference(prim, mData, atoms)
% Uses Material Studio / LAMMPS x-fastest ordering:
%   atom i → UC (ix = i_cell % dim_x,
%                iy = floor(i_cell/dim_x) % dim_y,
%                iz = floor(i_cell/(dim_x*dim_y)))
% so that phase factors exp(iq·R_n) are assigned to the correct lattice site.
    numAtoms   = size(atoms.pos, 1);
    typeList   = round(atoms.pos(:,1,1));
    atomsPerUC = prim.n_atoms;

    L         = [mData.Lx mData.Ly mData.Lz];
    dim_rough = max(1, round(L ./ prim.cell_diag));
    [~, imin] = min(dim_rough);
    prim_a    = L(imin) / dim_rough(imin);
    dim_box   = max(1, round(L / prim_a));

    dim_x = dim_box(1); dim_y = dim_box(2); dim_z = dim_box(3);
    ax = mData.Lx/dim_x; ay = mData.Ly/dim_y; az = mData.Lz/dim_z;

    switch mData.chainDir
        case 'x', N_UC_chain = dim_x;
        case 'y', N_UC_chain = dim_y;
        otherwise, N_UC_chain = dim_z;
    end

    total_UC  = numAtoms / atomsPerUC;
    numChains = max(1, round(total_UC / N_UC_chain));
    fprintf('[Reference] dim_box=%s | total_UC=%d | N_UC_chain=%d | numChains=%d\n', ...
        mat2str(dim_box), total_UC, N_UC_chain, numChains);

    if isempty(mData.masses)
        error('No masses found. Check .data file or add prim.xyz masses.');
    end

    % Material Studio x-fastest ordering: ix fastest, then iy, then iz
    i0     = (0:numAtoms-1)';
    i_cell = floor(i0 / atomsPerUC);
    b_atom = mod(i0, atomsPerUC) + 1;   % 1-based basis index

    ix = mod(i_cell, dim_x);
    iy = mod(floor(i_cell / dim_x), dim_y);
    iz = floor(i_cell / (dim_x * dim_y));

    % Ideal unit cell position R_n
    ref.R_n = [ix*ax, iy*ay, iz*az];   % [nAtoms × 3]

    t = min(typeList, numel(mData.masses));
    massList = mData.masses(t)';

    ref.basis    = b_atom;
    ref.masses   = massList;
    ref.N_UC     = N_UC_chain;
    ref.ax = ax; ref.ay = ay; ref.az = az;
    ref.chainDir = mData.chainDir;
end


% ── 4. Q-path ─────────────────────────────────────────────────────
function q_cart = make_q_path(ref, q_reduced)
    switch ref.chainDir
        case 'x', a=ref.ax; unit=[1 0 0];
        case 'y', a=ref.ay; unit=[0 1 0];
        otherwise, a=ref.az; unit=[0 0 1];
    end
    q_cart = q_reduced(:) * (2*pi/a) .* unit;
end


% ── Output helpers (CC 시스템과 동일한 패턴) ──────────────────────
function cfg = prepare_output_dir(cfg)
    sourceDir = fileparts(mfilename('fullpath'));
    if isempty(sourceDir), sourceDir = pwd; end
    cfg.sourceDir = sourceDir;

    if ~isfield(cfg, 'outputRoot') || isempty(cfg.outputRoot)
        cfg.outputRoot = fullfile(sourceDir, 'SED_outputs');
    end
    if ~isfield(cfg, 'runTag') || isempty(cfg.runTag)
        cfg.runTag = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    end
    cfg.outputDir = fullfile(cfg.outputRoot, ['SED_' cfg.runTag]);
    if ~exist(cfg.outputDir, 'dir'), mkdir(cfg.outputDir); end
    fprintf('Output folder: %s\n', cfg.outputDir);
end

function save_sed_result(mdata, cfg)
    dataFile = fullfile(cfg.outputDir, 'SED_data.mat');
    mdata.sourceDir = cfg.sourceDir;   % so PLOT.m can addpath back to main.m
    save(dataFile, 'mdata', '-v7.3');
    fprintf('Saved data: %s\n', dataFile);
end

function write_replot_script(outputDir)
    plotFile = fullfile(outputDir, 'PLOT.m');
    fid = fopen(plotFile, 'w');
    if fid < 0, error('Cannot write %s', plotFile); end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%% Replot saved SED — run this file from its own folder.\n');
    fprintf(fid, 'clear; close all;\n');
    fprintf(fid, 'thisDir = fileparts(mfilename(''fullpath''));\n');
    fprintf(fid, 'S = load(fullfile(thisDir, ''SED_data.mat''));\n');
    fprintf(fid, 'mdata = S.mdata;\n');
    fprintf(fid, 'if isfield(mdata, ''sourceDir'') && exist(mdata.sourceDir, ''dir'')\n');
    fprintf(fid, '    addpath(mdata.sourceDir);\n');
    fprintf(fid, 'end\n');
    fprintf(fid, 'cfg.outputDir = thisDir;\n');
    fprintf(fid, 'plot_SED(mdata, cfg);\n');
    fprintf('Wrote: %s\n', plotFile);
end


% ── 5. SED 계산 ───────────────────────────────────────────────────
function mdata = compute_SED(atoms, ref, q_cart, cfg)
% Dynasor-style SED: positive FFT bins, mass-weighted velocities,
% and normalization dt / (N_samples * N_UC * 2*pi).

    numSteps        = size(atoms.vel, 3);
    dt_fs           = cfg.timeStepFs;
    steps_per_split = floor(numSteps / cfg.numSplits);
    sim_time_ps     = dt_fs * steps_per_split * 1e-3;
    num_q           = size(q_cart, 1);
    n_freq          = floor(steps_per_split / 2) + 1;

    fprintf('[SED] dt=%.1ffs | steps=%d | T=%.1fps | %d q-pts\n', ...
        cfg.timeStepFs, steps_per_split, sim_time_ps, num_q);

    phase_fac = exp(1i * (ref.R_n * q_cart'));   % [nAtoms × nQ]
    vels      = atoms.vel(:, 2:4, 1:steps_per_split);
    basisVals = unique(ref.basis);
    num_basis = numel(basisVals);

    SED_x = zeros(n_freq, num_q);
    SED_y = zeros(n_freq, num_q);
    SED_z = zeros(n_freq, num_q);
    norm_factor = dt_fs / (steps_per_split * ref.N_UC * 2*pi);
    t0    = tic;

    for iq = 1:num_q
        acc_x = zeros(n_freq, 1);
        acc_y = zeros(n_freq, 1);
        acc_z = zeros(n_freq, 1);

        for ib = 1:num_basis
            idx   = find(ref.basis == ib);
            m_b   = ref.masses(idx(1));
            ph    = phase_fac(idx, iq);           % [nAtoms_b × 1]
            v3    = reshape(vels(idx,:,:), [numel(idx), 3, steps_per_split]);
            v_sum = squeeze(sum(v3 .* ph, 1));    % [3 × steps]
            v_fft = fft(v_sum, [], 2);
            v_fft = v_fft(:, 1:n_freq);

            acc_x = acc_x + m_b * abs(v_fft(1,:)).'.^2;
            acc_y = acc_y + m_b * abs(v_fft(2,:)).'.^2;
            acc_z = acc_z + m_b * abs(v_fft(3,:)).'.^2;
        end

        SED_x(:,iq) = acc_x * norm_factor;
        SED_y(:,iq) = acc_y * norm_factor;
        SED_z(:,iq) = acc_z * norm_factor;

        if mod(iq, 20) == 0 || iq == num_q
            elapsed = toc(t0);
            fprintf('  q %3d/%d  %.0fs  ETA %.0fs\n', ...
                iq, num_q, elapsed, elapsed/iq*(num_q-iq));
        end
    end

    mdata.SED_x    = SED_x;
    mdata.SED_y    = SED_y;
    mdata.SED_z    = SED_z;
    mdata.freq_THz = (0:n_freq-1)' / (steps_per_split * dt_fs) * 1e3;
    fprintf('[SED] Done in %.0fs.\n', toc(t0));
end


% ── 6. 시각화 ─────────────────────────────────────────────────────
function plot_SED(mdata, cfg)
    if nargin < 2 || ~isstruct(cfg), cfg = struct(); end
    if ~isfield(cfg, 'outputDir') || isempty(cfg.outputDir)
        cfg.outputDir = fileparts(mfilename('fullpath'));
        if isempty(cfg.outputDir), cfg.outputDir = pwd; end
    end
    if ~isfield(cfg, 'runTag') || isempty(cfg.runTag)
        tag = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    else
        tag = cfg.runTag;
    end

    qi    = mdata.q_reduced <= 0.5 + 1e-12;
    freq  = mdata.freq_THz(:);
    q_plt = 2 * mdata.q_reduced(qi);

    Sx = prepare_sed_intensity(mdata.SED_x(:, qi));
    Sy = prepare_sed_intensity(mdata.SED_y(:, qi));
    Sz = prepare_sed_intensity(mdata.SED_z(:, qi));
    St = prepare_sed_intensity(mdata.SED_x(:, qi) + mdata.SED_y(:, qi) + mdata.SED_z(:, qi));

    clim_x = safe_clim(Sx, [0 99]);
    clim_y = safe_clim(Sy, [0 99]);
    clim_z = safe_clim(Sz, [0 99]);
    clim_t = safe_clim(St, [0 99]);

    figure('Color','w','Position',[100 100 1800 500]);

    titles = {'SED_x', 'SED_y', 'SED_z', 'SED total'};
    datas  = {Sx, Sy, Sz, St};
    clims  = {clim_x, clim_y, clim_z, clim_t};
    cmaps  = {'turbo', 'turbo', 'turbo', paper_hot_colormap(256)};

    for k = 1:4
        ax = subplot(1, 4, k);
        pcolormesh_centers(ax, q_plt, freq, datas{k});
        set(ax, 'YDir','normal', 'Box','on', 'Layer','top', ...
            'FontSize', 11, 'LineWidth', 1.0, 'TickDir','in');
        axis(ax, 'tight');
        xlim(ax, [min(q_plt) max(q_plt)]);
        ylim(ax, [0 2]);
        xticks(ax, 0:0.2:1);
        clim(ax, clims{k});
        colormap(ax, cmaps{k});
        colorbar(ax);
        xlabel(ax, 'q (\pi/a)');
        ylabel(ax, 'Frequency (THz)');
        title(ax, titles{k});
    end

    saveName = fullfile(cfg.outputDir, sprintf('SED_%s.png', tag));
    exportgraphics(gcf, saveName, 'Resolution', 200);
    fprintf('Saved: %s\n', saveName);
end


function clim = safe_clim(Z, pct)
    vals = Z(:);
    vals = vals(isfinite(vals) & vals > 0);
    if isempty(vals), clim = [0 1]; return; end
    clim = prctile(vals, pct);
    if ~all(isfinite(clim)) || clim(1) >= clim(2)
        v = max(abs(vals));
        if ~isfinite(v) || v == 0, v = 1; end
        clim = [0 v];
    end
end

function I = prepare_sed_intensity(Z)
% Plot-only cleanup for paper-style SED visualization.
    I = max(Z, 0);
    I(~isfinite(I)) = 0;
    vals0 = I(isfinite(I) & I > 0);
    if ~isempty(vals0)
        I = max(I - prctile(vals0, 25), 0);
    end
    I = smooth_heatmap(I, 0.85, 0.50);
end

function cmap = paper_hot_colormap(n)
% Black-red-yellow-white map similar to current-correlation paper figures.
    if nargin < 1, n = 256; end
    x = [0.00 0.20 0.45 0.70 0.88 1.00];
    c = [0.00 0.00 0.00
         0.16 0.00 0.00
         0.75 0.00 0.00
         1.00 0.28 0.00
         1.00 0.90 0.00
         1.00 1.00 1.00];
    xi = linspace(0, 1, n);
    cmap = interp1(x, c, xi, 'linear');
end

function Zs = smooth_heatmap(Z, sigma_y, sigma_x)
% Small separable Gaussian smoothing for display only.
    if sigma_y <= 0 && sigma_x <= 0
        Zs = Z;
        return;
    end

    ky = gaussian_kernel_1d(sigma_y);
    kx = gaussian_kernel_1d(sigma_x);
    Zs = conv2(Z, ky(:), 'same');
    Zs = conv2(Zs, kx(:)', 'same');
end

function k = gaussian_kernel_1d(sigma)
    if sigma <= 0
        k = 1;
        return;
    end
    radius = max(1, ceil(3 * sigma));
    x = -radius:radius;
    k = exp(-(x.^2) / (2 * sigma^2));
    k = k / sum(k);
end

function h = pcolormesh_centers(ax, x, y, Z)
% MATLAB equivalent of matplotlib pcolormesh(x, y, Z, shading='auto')
% when x and y are cell centers.
    x = x(:)';
    y = y(:);
    if size(Z, 1) ~= numel(y) || size(Z, 2) ~= numel(x)
        error('pcolormesh_centers: Z must be [numel(y) x numel(x)].');
    end

    xe = centers_to_edges(x);
    ye = centers_to_edges(y');
    [X, Y] = meshgrid(xe, ye);

    C = zeros(numel(y) + 1, numel(x) + 1);
    C(1:end-1, 1:end-1) = Z;
    C(end, 1:end-1) = Z(end, :);
    C(1:end-1, end) = Z(:, end);
    C(end, end) = Z(end, end);

    h = surface(ax, X, Y, zeros(size(C)), C, ...
        'EdgeColor', 'none', 'FaceColor', 'flat');
    view(ax, 2);
end

function edges = centers_to_edges(centers)
    centers = centers(:)';
    if isscalar(centers)
        step = 0.5;
        edges = [centers - step, centers + step];
        return;
    end

    mid = 0.5 * (centers(1:end-1) + centers(2:end));
    first = centers(1) - 0.5 * (centers(2) - centers(1));
    last = centers(end) + 0.5 * (centers(end) - centers(end-1));
    edges = [first, mid, last];
end
