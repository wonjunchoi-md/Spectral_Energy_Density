%% ================================================================
%  Current Correlation Function — C_L(q,ω), C_T(q,ω)
%  dynasor 방식: ACF → FFT (Wiener-Khinchin)
%
%  [1] 타입별 partial current
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
%
%  메모리 절감: trajectory → matfile (디스크), 윈도우 프레임만 RAM 로드
% ================================================================

%% ===== Config ===================================================
cfg.folderPath = '';               % 비워두면 GUI 폴더 선택창
cfg.primFile   = 'prim_no_H.xyz';  % 선택한 폴더 안에 있어야 함
cfg.timeStepFs = 40;
cfg.maxSteps   = 0;
cfg.windowSize = 500;              % ACF 최대 time lag (# frames)
cfg.windowStep = 250;              % 윈도우 stride (50% overlap)

%% ===== Pipeline =================================================

[mf, mData, folderPath] = read_trajectory(cfg.folderPath, cfg.maxSteps);
prim      = read_prim_xyz(fullfile(folderPath, cfg.primFile));
ref       = build_reference(prim, mData);

q_reduced = (0 : floor(ref.N_UC/2)) / ref.N_UC;
q_cart    = make_q_path(ref, q_reduced);

[omega_THz, CL, CT] = compute_current_correlate(mf, q_cart, cfg);
plot_CC(CL, CT, omega_THz, q_reduced);


%% ================================================================
%  Local Functions
%% ================================================================

function [omega_THz, CL, CT] = compute_current_correlate(mf, q_cart, cfg)
% C_L(q,ω), C_T(q,ω) — 윈도우별 즉석 계산 (jL/jT 전체 미리 저장 안 함)

    Nt      = mf.nSteps;
    N_atoms = mf.nAtoms;
    dt      = cfg.timeStepFs * 1e-15;
    num_q   = size(q_cart, 1);

    if ~isfield(cfg, 'windowSize'), cfg.windowSize = floor(Nt/5); end
    if ~isfield(cfg, 'windowStep'), cfg.windowStep = floor(cfg.windowSize/2); end
    ws    = cfg.windowSize;
    wstep = cfg.windowStep;
    N_tc  = ws + 1;

    q_mag = sqrt(sum(q_cart.^2, 2));
    q_hat = q_cart ./ (q_mag + eps);

    % 타입 목록은 첫 프레임에서만 로드
    type_list  = round(mf.pos(:, 1, 1));
    atom_types = unique(type_list);
    n_types    = numel(atom_types);

    starts    = 1 : wstep : (Nt - ws);
    n_windows = numel(starts);
    CL_acf    = zeros(num_q, N_tc);
    CT_acf    = zeros(num_q, N_tc);

    fprintf('[CC] %d windows × %d lags × %d q-pts ...\n', n_windows, N_tc, num_q);
    t_acf = tic;

    for wi = 1:n_windows
        t0 = starts(wi);
        t1 = t0 + ws;

        % 이 윈도우 프레임만 디스크에서 로드
        pos_win = mf.pos(:, :, t0:t1);   % [N_atoms × 4 × (ws+1)]
        vel_win = mf.vel(:, :, t0:t1);

        % jL/jT 이 윈도우에서만 계산
        jL = zeros(num_q, ws+1, n_types, 'like', complex(0));
        jT = zeros(num_q, 3, ws+1, n_types, 'like', complex(0));
        for li = 1:ws+1
            r = pos_win(:, 2:4, li);
            v = vel_win(:, 2:4, li);
            for s = 1:n_types
                idx   = type_list == atom_types(s);
                phase = exp(1i * (r(idx,:) * q_cart'));
                j_s   = phase' * v(idx,:);
                jL_s  = sum(j_s .* q_hat, 2);
                jL(:, li, s)    = jL_s;
                jT(:, :, li, s) = j_s - jL_s .* q_hat;
            end
        end

        % ACF 누적
        for tau = 0:ws
            for s1 = 1:n_types
                for s2 = s1:n_types
                    dCL = real(jL(:,1,s1) .* conj(jL(:,tau+1,s2)));
                    dCT = 0.5 * real(sum(jT(:,:,1,s1) .* conj(jT(:,:,tau+1,s2)), 2));
                    if s1 ~= s2
                        dCL = dCL + real(jL(:,1,s2) .* conj(jL(:,tau+1,s1)));
                        dCT = dCT + 0.5*real(sum(jT(:,:,1,s2) .* conj(jT(:,:,tau+1,s1)), 2));
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

    CL_acf = CL_acf / (n_windows * N_atoms);
    CT_acf = CT_acf / (n_windows * N_atoms);

    CL_mir = [CL_acf, CL_acf(:, end:-1:2)];
    CT_mir = [CT_acf, CT_acf(:, end:-1:2)];
    N_fft  = 2*ws + 1;
    n_pos  = ws + 1;

    CL_psd = real(fft(CL_mir, [], 2)) * dt;
    CT_psd = real(fft(CT_mir, [], 2)) * dt;

    CL = CL_psd(:, 1:n_pos)';
    CT = CT_psd(:, 1:n_pos)';

    omega_THz = (0 : n_pos-1) / (N_fft * dt) / 1e12;
    fprintf('[CC] Done in %.0fs.\n', toc(t_acf));
end


function plot_CC(CL, CT, omega_THz, q_reduced)
    qi    = q_reduced <= 0.5 + 1e-12;
    q_plt = 2 * q_reduced(qi);

    figure('Color','w','Position',[100 100 1200 500]);

    ax1 = subplot(1,2,1);
    imagesc(q_plt, omega_THz, log(max(CL(:,qi), 0) + 1e-30));
    set(gca,'YDir','normal'); axis tight; ylim([0 1]);
    colormap(ax1, 'turbo'); colorbar;
    xlabel('q (π/a)'); ylabel('Frequency (THz)'); title('C_L (longitudinal)');

    ax2 = subplot(1,2,2);
    imagesc(q_plt, omega_THz, log(max(CT(:,qi), 0) + 1e-30));
    set(gca,'YDir','normal'); axis tight; ylim([0 1]);
    colormap(ax2, 'parula'); colorbar;
    xlabel('q (π/a)'); ylabel('Frequency (THz)'); title('C_T (transverse)');

    saveName = fullfile(fileparts(mfilename('fullpath')), ...
        sprintf('CC_%s.png', datestr(now,'yyyymmdd_HHMMSS')));
    exportgraphics(gcf, saveName, 'Resolution', 200);
    fprintf('Saved: %s\n', saveName);
end


function [mf, mData, folderPath] = read_trajectory(folderPath, maxSteps)
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

    dumpPath = fullfile(folderPath, dumpFiles(1).name);
    matPath  = fullfile(folderPath, 'trajectory_cache.mat');

    if exist(matPath, 'file')
        fprintf('Using cached trajectory: %s\n', matPath);
        mf = matfile(matPath, 'Writable', false);
    else
        mf = stream_dump_to_matfile(dumpPath, matPath, maxSteps);
    end

    mData = parse_box_from_dump(dumpPath);
    if ~isempty(dataFiles)
        tmp = parse_data_file(fullfile(folderPath, dataFiles(1).name));
        mData.masses = tmp.masses;
    end
    fprintf('Trajectory: %d atoms × %d steps\n', mf.nAtoms, mf.nSteps);
end


function mf = stream_dump_to_matfile(dumpPath, matPath, maxSteps)
% dump 파일을 프레임별로 스트리밍 → matfile 저장 (RAM에 프레임 하나만 올라옴)

    % Pass 1: nAtoms, nSteps 파악 (데이터 파싱 없이 헤더만 스캔)
    fprintf('[Stream] Scanning %s ...\n', dumpPath);
    fid = fopen(dumpPath, 'r');
    nAtoms = 0; nSteps = 0;
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line), continue; end
        line = strtrim(line);
        if contains(line, 'NUMBER OF ATOMS') && nAtoms == 0
            nAtoms = str2double(strtrim(fgetl(fid)));
        elseif contains(line, 'ITEM: TIMESTEP')
            nSteps = nSteps + 1;
            if maxSteps > 0 && nSteps >= maxSteps, break; end
        end
    end
    fclose(fid);
    if maxSteps > 0, nSteps = min(nSteps, maxSteps); end
    fprintf('[Stream] %d atoms × %d frames\n', nAtoms, nSteps);

    % matfile 사전 할당 (디스크에 공간 예약)
    fprintf('[Stream] Preallocating %s ...\n', matPath);
    mf = matfile(matPath, 'Writable', true);
    mf.nAtoms = nAtoms;
    mf.nSteps = nSteps;
    mf.pos    = zeros(nAtoms, 4, nSteps);   % [type x y z]
    mf.vel    = zeros(nAtoms, 4, nSteps);   % [type vx vy vz]

    % Pass 2: 프레임별 스트리밍 저장
    fprintf('[Stream] Writing frames ...\n');
    fid = fopen(dumpPath, 'r');
    step = 0; t0 = tic;
    while ~feof(fid)
        line = fgetl(fid);
        if ~ischar(line), continue; end
        if contains(strtrim(line), 'ITEM: ATOMS')
            step = step + 1;
            if maxSteps > 0 && step > maxSteps, break; end
            data = textscan(fid, '%f %f %f %f %f %f %f %f', nAtoms);
            mf.pos(:, :, step) = [data{2}, data{3}, data{4}, data{5}];
            mf.vel(:, :, step) = [data{2}, data{6}, data{7}, data{8}];
            if mod(step, 50) == 0
                fprintf('  frame %d/%d  (%.0fs)\n', step, nSteps, toc(t0));
            end
        end
    end
    fclose(fid);
    fprintf('[Stream] Done in %.0fs → %s\n', toc(t0), matPath);
    mf = matfile(matPath, 'Writable', false);
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

function ref = build_reference(prim, mData)
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
