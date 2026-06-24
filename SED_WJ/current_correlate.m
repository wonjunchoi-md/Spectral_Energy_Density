%% ================================================================
%  Current Correlation Function — C_L(q,ω), C_T(q,ω)
%  dynasor 방식: ACF → FFT (Wiener-Khinchin)
%
%  [1] 타입별 partial currentc
%      j_s(q,t) = Σ_{i∈s} exp(iq·r_i(t)) · v_i(t)
%      j_L_s    = j_s · q̂           (종파)
%      j_T_s    = j_s − j_L_s · q̂  (횡파)
%
%  [2] ACF (윈도우 평균)
%      C_L(q,τ) = (1/N) Σ_{s1,s2} Re[ j_L_s1(q,t)·j*_L_s2(q,t+τ) ]_t
%
%  [3] Wiener-Khinchin: PSD = dt · Re[ FFT[ mirror(ACF) ] ]
%
%  [4] Per-atom 정규화: / N_atoms
% ================================================================

%% ===== Config ===================================================
cfg.folderPath = '';               % 비워두면 GUI 폴더 선택창
cfg.primFile   = 'prim_no_H.xyz';  % 선택한 폴더 안에 있어야 함
cfg.timeStepFs = 10;
cfg.maxSteps   = 10000;
cfg.windowSize = 10000;              % ACF 최대 time lag (# frames)
cfg.windowStep = 10000;              % 윈도우 stride (50% overlap)

%% ===== Pipeline =================================================

[atoms, mData, folderPath] = read_trajectory(cfg.folderPath, cfg.maxSteps);
prim      = read_prim_xyz(fullfile(folderPath, cfg.primFile));
ref       = build_reference(prim, mData, atoms);
q_reduced = (0 : floor(ref.N_UC/2)) / ref.N_UC;
q_cart    = make_q_path(ref, q_reduced);

[omega_THz, CL, CT] = compute_current_correlate(atoms, q_cart, cfg);
%% Plot
plot_CC(CL, CT, omega_THz, q_reduced);


%% ================================================================
%  Local Functions
%% ================================================================
function plot_CC(CL, CT, omega_THz, q_reduced)
    qi    = q_reduced <= 0.5 + 1e-12;
    q_plt = 2 * q_reduced(qi);

    % q 범위에 해당하는 데이터만 추출
    CLi = CL(:, qi);
    CTi = CT(:, qi);

    % NaN, Inf 제거
    CLi(~isfinite(CLi)) = 0;
    CTi(~isfinite(CTi)) = 0;

    % log scale 변환
    ZL = log10(max(CLi, 1e-30));
    ZT = log10(max(CTi, 1e-30));

    % color limit 안전 계산
    % [2 98]보다 더 자르고 싶으면 [5 95] 또는 [10 90] 사용
    climL = safe_clim(ZL, [0.1 99.7]);
    climT = safe_clim(ZT, [0.1 99.7]);

    figure('Color','w','Position',[100 100 1200 500]);

    base = hot(256);
    cmap = base(80:end, :);   % remove black/dark part

    ax1 = subplot(1,2,1);
    imagesc(q_plt, omega_THz, ZL);
    set(ax1, 'YDir', 'normal');
    axis(ax1, 'tight');
    ylim(ax1, [0 1]);
    colormap(ax1, cmap);
    colorbar(ax1);
    caxis(ax1, climL);
    xlabel(ax1, 'q (\pi/a)');
    ylabel(ax1, 'Frequency (THz)');
    title(ax1, 'C_L (longitudinal)');

    ax2 = subplot(1,2,2);
    imagesc(q_plt, omega_THz, ZT);
    set(ax2, 'YDir', 'normal');
    axis(ax2, 'tight');
    ylim(ax2, [0 1]);
    colormap(ax2, cmap);
    colorbar(ax2);
    caxis(ax2, climT);
    xlabel(ax2, 'q (\pi/a)');
    ylabel(ax2, 'Frequency (THz)');
    title(ax2, 'C_T (transverse)');

    saveName = fullfile(fileparts(mfilename('fullpath')), ...
        sprintf('CC_%s.png', datestr(now,'yyyymmdd_HHMMSS')));

    exportgraphics(gcf, saveName, 'Resolution', 200);
    fprintf('Saved: %s\n', saveName);
end


function clim = safe_clim(Z, pct)
    vals = Z(:);
    vals = vals(isfinite(vals));

    if isempty(vals)
        clim = [-30 0];
        return;
    end

    clim = prctile(vals, pct);

    if ~all(isfinite(clim)) || clim(1) >= clim(2)
        v = median(vals, 'omitnan');

        if ~isfinite(v)
            v = 0;
        end

        clim = [v - 1, v + 1];
    end
end




function [omega_THz, CL, CT] = compute_current_correlate(atoms, q_cart, cfg)
% C_L(q,ω), C_T(q,ω)  —  dynasor 방식 (ACF → FFT, per-atom, partial)

    Nt      = size(atoms.vel, 3);
    dt      = cfg.timeStepFs;            % [fs]  — dynasor 단위계 통일
    num_q   = size(q_cart, 1);
    N_atoms = size(atoms.vel, 1);

    if ~isfield(cfg, 'windowSize'), cfg.windowSize = floor(Nt/5); end
    if ~isfield(cfg, 'windowStep'), cfg.windowStep = floor(cfg.windowSize/2); end
    ws    = cfg.windowSize;
    wstep = cfg.windowStep;
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

    % ── Wiener-Khinchin: mirror ACF → FFT → real ───────────────
    % ACF는 짝함수: mirror [C(0)..C(ws), C(ws-1)..C(1)] → 길이 2ws+1
    CL_mir = [CL_acf, CL_acf(:, end:-1:2)];   % [Nq × (2ws+1)]
    CT_mir = [CT_acf, CT_acf(:, end:-1:2)];
    N_fft  = 2*ws + 1;
    n_pos  = ws + 1;   % rfft 양수 주파수 수 = floor(N_fft/2)+1

    CL_psd = real(fft(CL_mir, [], 2)) * dt;   % [Nq × N_fft]
    CT_psd = real(fft(CT_mir, [], 2)) * dt;

    CL = CL_psd(:, 1:n_pos)';   % [N_pos × Nq]
    CT = CT_psd(:, 1:n_pos)';

    omega_THz = (0 : n_pos-1) / (N_fft * dt) * 1e3;    % 선형 주파수 [THz]  (dt [fs] → *1e3)
    fprintf('[CC] Done in %.0fs.\n', toc(t_acf));
end


<<<<<<< Updated upstream
=======
function plot_CC(CL, CT, omega_THz, q_reduced)
    % omega_THz 는 이미 양수 주파수만 포함 (0 ~ Nyquist)
    qi      = q_reduced <= 0.5 + 1e-12;
    q_plt   = 2 * q_reduced(qi);
    CL_qi   = CL(:, qi);
    CT_qi   = CT(:, qi);
    diff_qi = CL_qi - CT_qi;

    freq_max  = max(omega_THz);

    % 99th-percentile clipping (linear scale)
    vmax_L    = prctile(CL_qi(:),       99);
    vmax_T    = prctile(CT_qi(:),       99);
    vlim_diff = prctile(abs(diff_qi(:)), 99);

    % blue-white-red diverging colormap for C_L − C_T
    n_cmap = 256; half = n_cmap / 2;
    r = [linspace(0,1,half), ones(1,half)];
    g = [linspace(0,1,half), linspace(1,0,half)];
    b = [ones(1,half),       linspace(1,0,half)];
    cmap_bwr = [r; g; b]';

    figure('Color','w','Position',[100 100 1800 500]);

    ax1 = subplot(1,3,1);
    imagesc(q_plt, omega_THz, CL_qi);
    set(gca,'YDir','normal'); axis tight; ylim([0 freq_max]);
    caxis([0 vmax_L]);
    colormap(ax1, 'turbo'); colorbar;
    xlabel('q (π/a)'); ylabel('Frequency (THz)'); title('C_L (longitudinal)');

    ax2 = subplot(1,3,2);
    imagesc(q_plt, omega_THz, CT_qi);
    set(gca,'YDir','normal'); axis tight; ylim([0 freq_max]);
    caxis([0 vmax_T]);
    colormap(ax2, 'parula'); colorbar;
    xlabel('q (π/a)'); ylabel('Frequency (THz)'); title('C_T (transverse)');

    ax3 = subplot(1,3,3);
    imagesc(q_plt, omega_THz, diff_qi);
    set(gca,'YDir','normal'); axis tight; ylim([0 freq_max]);
    caxis([-vlim_diff vlim_diff]);
    colormap(ax3, cmap_bwr); colorbar;
    xlabel('q (π/a)'); ylabel('Frequency (THz)'); title('C_L − C_T');

    saveName = fullfile(fileparts(mfilename('fullpath')), ...
        sprintf('CC_%s.png', datestr(now,'yyyymmdd_HHMMSS')));
    exportgraphics(gcf, saveName, 'Resolution', 200);
    fprintf('Saved: %s\n', saveName);
end


>>>>>>> Stashed changes
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
