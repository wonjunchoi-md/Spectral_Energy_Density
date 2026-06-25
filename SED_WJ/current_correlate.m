%% ================================================================
%  Current Correlation Function — C_L(q,ω), C_T(q,ω)
%  dynasor 방식: ACF → Filon cosine transform
% ================================================================
%% ===== Config ===================================================
cfg.folderPath = '';                 % 비워두면 GUI 폴더 선택창
cfg.primFile   = 'prim_no_H.xyz';    % 선택한 폴더 안에 있어야 함
cfg.timeStepFs = 40;
cfg.maxSteps   = 0;
cfg.windowSize = 12501;              % ACF 최대 time lag (# frames)
cfg.windowStep = 12501;              % 윈도우 stride (50% overlap)
cfg.plotClipPercentile = 99;      % plot color clipping percentile

%% ===== Pipeline =================================================

[atoms, mData, folderPath] = read_trajectory(cfg.folderPath, cfg.maxSteps);
prim      = read_prim_xyz(fullfile(folderPath, cfg.primFile));
ref       = build_reference(prim, mData, atoms);
q_reduced = (0 : floor(ref.N_UC/2)) / ref.N_UC;
q_cart    = make_q_path(ref, q_reduced);

[omega_THz, CL, CT] = compute_current_correlate(atoms, q_cart, cfg);
%% Plot
plot_CC(CL, CT, omega_THz, q_reduced, cfg);


%% ================================================================
function plot_CC(CL, CT, omega_THz, q_reduced, cfg)
    % Paper-style current intensity plots with a black-red-yellow-white map.
    if nargin < 5 || ~isfield(cfg, 'plotClipPercentile')
        cfg.plotClipPercentile = 99.5;
    end
    qi    = q_reduced <= 0.5 + 1e-12;
    q_plt = 2 * q_reduced(qi);

    CLi  = CL(:, qi);  CTi = CT(:, qi);
    CLi(~isfinite(CLi)) = 0;
    CTi(~isfinite(CTi)) = 0;
    freq_max = max(omega_THz);

    CLp = prepare_current_intensity(CLi);
    CTp = prepare_current_intensity(CTi);
    Cp  = prepare_current_intensity(max(CLi, 0) + max(CTi, 0));

    vals = [CLp(:); CTp(:); Cp(:)];
    vals = vals(isfinite(vals) & vals > 0);
    if isempty(vals)
        clim_i = [0 1];
    else
        vmax = prctile(vals, cfg.plotClipPercentile);
        if ~isfinite(vmax) || vmax <= 0, vmax = max(vals); end
        if ~isfinite(vmax) || vmax <= 0, vmax = 1; end
        clim_i = [0 vmax];
    end

    fig = figure('Color','w','Position',[80 100 1320 430]);
    tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile(tl, 1);
    plot_current_panel(ax1, q_plt, omega_THz, CLp, clim_i, 'C_L');
    ylabel(ax1, 'Frequency (THz)');

    ax2 = nexttile(tl, 2);
    plot_current_panel(ax2, q_plt, omega_THz, CTp, clim_i, 'C_T');
    ylabel(ax2, '');
    ax2.YTickLabel = [];

    ax3 = nexttile(tl, 3);
    plot_current_panel(ax3, q_plt, omega_THz, Cp, clim_i, 'C_L + C_T');
    ylabel(ax3, '');
    ax3.YTickLabel = [];

    set([ax1 ax2 ax3], 'YLim', [0 freq_max]);
    cmap = paper_hot_colormap(256);
    colormap(ax1, cmap);
    colormap(ax2, cmap);
    colormap(ax3, cmap);
    cb = colorbar(ax3);
    cb.Layout.Tile = 'east';
    cb.Label.String = 'C(q,\omega)';
    cb.Label.FontSize = 11;

    saveName = fullfile(fileparts(mfilename('fullpath')), ...
        sprintf('CC_%s.png', char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))));
    exportgraphics(gcf, saveName, 'Resolution', 200);
    fprintf('Saved: %s\n', saveName);
end

function I = prepare_current_intensity(Z)
% Plot-only cleanup: keep positive intensity, darken background, smooth speckle.
    I = max(Z, 0);
    I(~isfinite(I)) = 0;
    vals0 = I(isfinite(I) & I > 0);
    if ~isempty(vals0)
        I = max(I - prctile(vals0, 35), 0);
    end
    I = smooth_heatmap(I, 0.9, 0.55);
end

function plot_current_panel(ax, q_plt, omega_THz, Z, clim_i, panel_title)
    pcolormesh_centers(ax, q_plt, omega_THz, Z);
    set(ax, 'YDir', 'normal', 'Box', 'on', 'Layer', 'top', ...
        'FontSize', 12, 'LineWidth', 1.0, 'TickDir', 'in');
    axis(ax, 'tight');
    xlim(ax, [min(q_plt) max(q_plt)]);
    xticks(ax, 0:0.2:1);
    clim(ax, clim_i);
    xlabel(ax, 'q (\pi/a)');
    title(ax, panel_title);
end


function cmap = paper_hot_colormap(n)
% Black-red-yellow map with a muted high end to avoid white saturation.
    if nargin < 1, n = 256; end
    x = [0.00 0.20 0.45 0.70 0.90 1.00];
    c = [0.00 0.00 0.00
         0.16 0.00 0.00
         0.75 0.00 0.00
         1.00 0.28 0.00
         1.00 0.82 0.00
         1.00 0.94 0.42];
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




function [omega_THz, CL, CT] = compute_current_correlate(atoms, q_cart, cfg)
% C_L(q,ω), C_T(q,ω)  —  dynasor 방식 (ACF → Filon, per-atom, partial)

    Nt      = size(atoms.vel, 3);
    dt      = cfg.timeStepFs;            % [fs]  — dynasor 단위계 통일
    num_q   = size(q_cart, 1);
    N_atoms = size(atoms.vel, 1);

    if ~isfield(cfg, 'windowSize'), cfg.windowSize = floor(Nt/5); end
    if ~isfield(cfg, 'windowStep'), cfg.windowStep = floor(cfg.windowSize/2); end
    ws    = cfg.windowSize;
    wstep = cfg.windowStep;
    if mod(ws, 2) ~= 0
        error('cfg.windowSize must be even to match dynasor fourier_cos_filon.');
    end
    N_tc  = ws + 1;   % lag τ = 0, 1, ..., ws

    q_mag = sqrt(sum(q_cart.^2, 2));
    q_hat = zeros(size(q_cart));

    nonzero_q = q_mag > 0;
    q_hat(nonzero_q,:) = q_cart(nonzero_q,:) ./ q_mag(nonzero_q);

    % ── 원자 타입 분류 ──────────────────────────────────────────
    type_list  = round(atoms.pos(:, 1, 1));
    atom_types = unique(type_list);
    n_types    = numel(atom_types);

    % ── [1] Step 1: 타입별 j_L, j_T 사전 계산 ─────────────────
    % jL{s}: [Nq × Nt]  complex
    % jT{s}: [Nq × 3 × Nt]  complex
    fprintf('[CC] Pre-computing j by type: %d types × %d frames ...\n', n_types, Nt);
    jL = cell(n_types, 1);
    jT = cell(n_types, 1);
    for s = 1:n_types
        jL{s} = zeros(num_q, Nt, 'like', complex(0));
        jT{s} = zeros(num_q, 3, Nt, 'like', complex(0));
    end

    t_pre = tic;
    for it = 1:Nt
        r = atoms.pos(:, 2:4, it);
        v = atoms.vel(:, 2:4, it);
        for s = 1:n_types
            idx   = type_list == atom_types(s);
            phase = exp(1i * (r(idx,:) * q_cart'));   % [Ns × Nq]
            j_s   = phase' * v(idx,:);                % [Nq × 3]
            jL_s  = sum(j_s .* q_hat, 2);             % [Nq × 1]
            jL{s}(:, it)    = jL_s;
            jT{s}(:, :, it) = j_s - jL_s .* q_hat;
        end
        if mod(it, 1000) == 0
            fprintf('  pre-compute %d/%d  (%.0fs)\n', it, Nt, toc(t_pre));
        end
    end
    fprintf('[CC] Pre-compute done in %.0fs.\n', toc(t_pre));

    % ── [2] Step 2: 윈도우 기반 ACF 누적 ──────────────────────
    starts    = 1 : wstep : (Nt - ws);
    n_windows = numel(starts);
    if n_windows == 0
        error('No windows processed. Nt (%d) must be larger than cfg.windowSize (%d).', Nt, ws);
    end
    CL_acf    = zeros(num_q, N_tc);
    CT_acf    = zeros(num_q, N_tc);

    fprintf('[CC] ACF: %d windows × %d lags × %d q-pts ...\n', n_windows, N_tc, num_q);
    t_acf = tic;

    for wi = 1:n_windows
        t0 = starts(wi);
        for tau = 0:ws
            t_tau = t0 + tau;
            for s1 = 1:n_types
                for s2 = s1:n_types
                    dCL = real(jL{s1}(:, t0) .* conj(jL{s2}(:, t_tau)));
                    dCT = 0.5 * real(sum(jT{s1}(:,:,t0) .* conj(jT{s2}(:,:,t_tau)), 2));
                    if s1 ~= s2   % (s2,s1) 대칭항 추가
                        dCL = dCL + real(jL{s2}(:, t0) .* conj(jL{s1}(:, t_tau)));
                        dCT = dCT + 0.5 * real(sum(jT{s2}(:,:,t0) .* conj(jT{s1}(:,:,t_tau)), 2));
                    end
                    CL_acf(:, tau+1) = CL_acf(:, tau+1) + dCL;
                    CT_acf(:, tau+1) = CT_acf(:, tau+1) + dCT;
                end
            end
        end
        if mod(wi, max(1, floor(n_windows/10))) == 0
            fprintf('  window %d/%d  (%.0fs)\n', wi, n_windows, toc(t_acf));
        end
    end

    % ── [3][4] Per-atom 정규화 + 윈도우 평균 ───────────────────
    CL_acf = CL_acf / (n_windows * N_atoms);
    CT_acf = CT_acf / (n_windows * N_atoms);

    % Dynasor uses fourier_cos_filon(C(q,t), dt), not mirrored FFT.
    [omega_rad_fs, CL_qw] = fourier_cos_filon_matlab(CL_acf, dt);
    [~,            CT_qw] = fourier_cos_filon_matlab(CT_acf, dt);

    CL = CL_qw';   % [N_freq x Nq]
    CT = CT_qw';

    omega_THz = omega_rad_fs(:) / (2*pi) * 1e3;  % rad/fs -> cycles/ps = THz
    fprintf('[CC] Done in %.0fs.\n', toc(t_acf));
end

function [omega, F] = fourier_cos_filon_matlab(f, dt)
% MATLAB port of dynasor.post_processing.filon.fourier_cos_filon.
% f is [Nq x Nt] with Nt = window_size + 1 and window_size even.
    if ndims(f) ~= 2
        error('fourier_cos_filon_matlab: f must be a 2D array.');
    end

    [n_rows, Nt] = size(f);
    if mod(Nt, 2) == 0 || Nt < 3
        error('fourier_cos_filon_matlab: f must have an odd number of time samples >= 3.');
    end

    omega = linspace(0, pi / dt, Nt);  % rad/fs, same grid as dynasor
    time = (0:Nt-1)' * dt;
    F = zeros(n_rows, Nt);

    block_size = 256;
    for i0 = 1:block_size:Nt
        i1 = min(Nt, i0 + block_size - 1);
        wb = omega(i0:i1);
        [alpha, beta, gamma] = filon_alpha_beta_gamma(wb * dt);

        weights = cos(time * wb);
        weights(1:2:end, :) = weights(1:2:end, :) .* beta;
        weights(2:2:end-1, :) = weights(2:2:end-1, :) .* gamma;

        weights(1, :) = 0.5 * weights(1, :) - alpha .* sin(wb * time(1));
        weights(end, :) = 0.5 * weights(end, :) + alpha .* sin(wb * time(end));

        F(:, i0:i1) = 2 * dt * (f * weights);
    end
end

function [alpha, beta, gamma] = filon_alpha_beta_gamma(theta)
% Vectorized coefficients from dynasor.post_processing.filon._alpha_beta_gamma_single.
    alpha = zeros(size(theta));
    beta = zeros(size(theta));
    gamma = zeros(size(theta));

    zero = abs(theta) < eps;
    alpha(zero) = 0.0;
    beta(zero) = 2/3;
    gamma(zero) = 4/3;

    nz = ~zero;
    th = theta(nz);
    s = sin(th);
    c = cos(th);

    alpha(nz) = (th.^2 + th .* s .* c - 2 * s.^2) ./ th.^3;
    beta(nz) = 2 * (th .* (1 + c.^2) - 2 * s .* c) ./ th.^3;
    gamma(nz) = 4 * (s - th .* c) ./ th.^3;
end

% ── 아래는 main.m 과 동일한 헬퍼 함수들 ──────────────────────────

function [atoms, mData, folderPath] = read_trajectory(folderPath, maxSteps)
    if isempty(folderPath)
        folderPath = uigetdir(pwd, 'Select data folder');
        if isequal(folderPath, 0), folderPath = pwd; end
    end
    if nargin < 2, maxSteps = 0; end

    dumpFiles = dir(fullfile(folderPath, '*.lammpstrj'));
    dumpFiles = dumpFiles(~startsWith({dumpFiles.name}, '.'));
    if isempty(dumpFiles), error('No *.lammpstrj found in %s', folderPath); end
    dataFiles = dir(fullfile(folderPath, '*.data'));
    if isempty(dataFiles), dataFiles = dir(fullfile(folderPath, 'data.*')); end

    fprintf('Reading %s ...\n', dumpFiles(1).name);
    out = read_combined_dump(fullfile(folderPath, dumpFiles(1).name), maxSteps);
    atoms.pos = out.pos; atoms.vel = out.vel;
    atoms.numAtoms = out.nAtoms; atoms.numSteps = out.nSteps;

    mData = parse_box_from_dump(fullfile(folderPath, dumpFiles(1).name));
    if ~isempty(dataFiles)
        tmp = parse_data_file(fullfile(folderPath, dataFiles(1).name));
        mData.masses = tmp.masses;
    end
    fprintf('Loaded: %d atoms × %d steps\n', atoms.numAtoms, atoms.numSteps);
end

function out = read_combined_dump(filename, maxSteps)
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
            pos_cells{step} = [data{1}, data{3}, data{4}, data{5}];   % [type x y z]
            vel_cells{step} = [data{1}, data{6}, data{7}, data{8}];   % [type vx vy vz]
            if mod(step, 50) == 0
                fprintf('  frame %d  (%.0fs elapsed)\n', step, toc(t0));
            end
        end
    end
    fclose(fid);
    fprintf('  total %d frames read in %.0fs\n', step, toc(t0));
    nSteps = numel(pos_cells);
    pos = zeros(nAtoms, 4, nSteps); vel = zeros(nAtoms, 4, nSteps);
    for s = 1:nSteps, pos(:,:,s) = pos_cells{s}; vel(:,:,s) = vel_cells{s}; end
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
end

function mData = parse_data_file(dataPath)
    fid = fopen(dataPath, 'r');
    C = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', ''); C = C{1};
    fclose(fid);
    mData = struct('masses', []);
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
end

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
end

function ref = build_reference(prim, mData, atoms)
    atomsPerUC = prim.n_atoms;
    L = [mData.Lx mData.Ly mData.Lz];
    dim_rough = max(1, round(L ./ prim.cell_diag));
    [~, imin] = min(dim_rough);
    prim_a = L(imin) / dim_rough(imin);
    dim_box = max(1, round(L / prim_a));
    switch mData.chainDir
        case 'x', Nz_chain = dim_box(1);
        case 'y', Nz_chain = dim_box(2);
        otherwise, Nz_chain = dim_box(3);
    end
    fprintf('[Reference] dim_box=%s | Nz=%d\n', mat2str(dim_box), Nz_chain);
    ax = mData.Lx/dim_box(1); ay = mData.Ly/dim_box(2); az = mData.Lz/dim_box(3);
    ref.N_UC     = Nz_chain;
    ref.ax = ax; ref.ay = ay; ref.az = az;
    ref.chainDir = mData.chainDir;
end

function q_cart = make_q_path(ref, q_reduced)
    switch ref.chainDir
        case 'x', a=ref.ax; unit=[1 0 0];
        case 'y', a=ref.ay; unit=[0 1 0];
        otherwise, a=ref.az; unit=[0 0 1];
    end
    q_cart = q_reduced(:) * (2*pi/a) .* unit;
end
