%% ================================================================
%  SED Analysis — main.m
% ================================================================

%% ===== Config ===================================================
cfg.folderPath = 'SED_Align';
cfg.primFile   = 'SED_Align/prim_no_H.xyz';
cfg.timeStepFs = 40;       % dump 간격 [fs]  (dt=0.5fs × dump_every=80)
cfg.numSplits  = 1;
cfg.maxSteps   = 0;        % 읽을 최대 프레임 수 (0 = 전체)
% cfg.qPoints 는 없음 — build_reference 후 N_UC 에서 자동 결정

%% ===== Pipeline =================================================

[atoms, mData] = read_trajectory(cfg.folderPath, cfg.maxSteps);
prim           = read_prim_xyz(cfg.primFile);
ref            = build_reference(prim, mData, atoms);
% BvK q-points: n=0..N/2  (총 N/2+1 개, 물리적으로 의미있는 유일한 점들)
q_reduced      = (0 : floor(ref.N_UC/2)) / ref.N_UC;
q_cart         = make_q_path(ref, q_reduced);
[freq_THz, SED_x, SED_y, SED_z] = compute_SED(atoms, ref, q_cart, cfg);
plot_SED(SED_x, SED_y, SED_z, freq_THz, q_reduced);


%% ================================================================
%  Local Functions
%% ================================================================

% ── 1. Trajectory reader ─────────────────────────────────────────
function [atoms, mData] = read_trajectory(folderPath, maxSteps)
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
    numAtoms   = size(atoms.pos, 1);
    typeList   = round(atoms.pos(:,1,1));
    atomsPerUC = prim.n_atoms;

    % prim cell 재스케일: 셀 수가 가장 적은 축에서 prim_a 역산
    L         = [mData.Lx mData.Ly mData.Lz];
    dim_rough = max(1, round(L ./ prim.cell_diag));
    [~, imin] = min(dim_rough);
    prim_a    = L(imin) / dim_rough(imin);
    dim_box   = max(1, round(L / prim_a));

    dim_x = dim_box(1); dim_y = dim_box(2); dim_z = dim_box(3);
    ax = mData.Lx/dim_x; ay = mData.Ly/dim_y; az = mData.Lz/dim_z;

    switch mData.chainDir
        case 'x', Nz_chain = dim_x;
        case 'y', Nz_chain = dim_y;
        otherwise, Nz_chain = dim_z;
    end

    total_UC  = numAtoms / atomsPerUC;
    numChains = max(1, round(total_UC / Nz_chain));
    fprintf('[Reference] dim_box=%s | total_UC=%d | Nz=%d | numChains=%d\n', ...
        mat2str(dim_box), total_UC, Nz_chain, numChains);

    if isempty(mData.masses)
        error('No masses found. Check .data file or add prim.xyz masses.');
    end

    % 원자별 (chainIdx, ucInChain, basisInUC) 벡터화 계산
    atomsPerChain = round(numAtoms / numChains);
    i0            = (0:numAtoms-1)';
    chainIdx      = floor(i0 / atomsPerChain) + 1;
    idxInChain    = mod(i0, atomsPerChain);
    ucInChain     = floor(idxInChain / atomsPerUC) + 1;
    basisInUC     = mod(idxInChain, atomsPerUC) + 1;
    basisIndex    = (chainIdx-1)*atomsPerUC + basisInUC;

    t = min(typeList, numel(mData.masses));
    massList = mData.masses(t)';

    % R_n: 이상 격자 위치 [nAtoms × 3]
    switch mData.chainDir
        case 'x', Np=dim_y; da=ay; db=az;
        case 'y', Np=dim_x; da=ax; db=az;
        otherwise, Np=dim_x; da=ax; db=ay;
    end
    ip = mod(chainIdx-1, Np);
    jp = floor((chainIdx-1)/Np);
    iz = ucInChain - 1;

    switch mData.chainDir
        case 'x', ref.R_n = [iz*ax, ip*da, jp*db];
        case 'y', ref.R_n = [ip*da, iz*ay, jp*db];
        otherwise, ref.R_n = [ip*da, jp*db, iz*az];
    end

    ref.basis    = mod(basisIndex-1, atomsPerUC) + 1;
    ref.masses   = massList;
    ref.N_UC     = Nz_chain;
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


% ── 5. SED 계산 ───────────────────────────────────────────────────
function [freq_THz, SED_x, SED_y, SED_z] = compute_SED(atoms, ref, q_cart, cfg)
% SED(q,ω) = (1/4πT·N) Σ_b m_b |Σ_n exp(iq·R_n) FFT[v_{b,n}(t)]|²

    numSteps        = size(atoms.vel, 3);
    dt              = cfg.timeStepFs * 1e-15;
    steps_per_split = floor(numSteps / cfg.numSplits);
    sim_time_ps     = dt * steps_per_split * 1e12;
    num_q           = size(q_cart, 1);

    fprintf('[SED] dt=%.1ffs | steps=%d | T=%.1fps | %d q-pts\n', ...
        cfg.timeStepFs, steps_per_split, sim_time_ps, num_q);

    phase_fac = exp(1i * (ref.R_n * q_cart'));   % [nAtoms × nQ]
    vels      = atoms.vel(:, 2:4, 1:steps_per_split);
    basisVals = unique(ref.basis);
    num_basis = numel(basisVals);

    SED_x = zeros(steps_per_split, num_q);
    SED_y = zeros(steps_per_split, num_q);
    SED_z = zeros(steps_per_split, num_q);
    norm  = 4 * pi * sim_time_ps * ref.N_UC;
    t0    = tic;

    for iq = 1:num_q
        acc_x = zeros(steps_per_split, 1);
        acc_y = zeros(steps_per_split, 1);
        acc_z = zeros(steps_per_split, 1);

        for ib = 1:num_basis
            idx   = find(ref.basis == ib);
            m_b   = ref.masses(idx(1));
            ph    = phase_fac(idx, iq);           % [nAtoms_b × 1]
            v3    = reshape(vels(idx,:,:), [numel(idx), 3, steps_per_split]);
            % Σ_n  exp(iq·R_n) · v_{b,n}(t)
            v_sum = squeeze(sum(v3 .* ph, 1));    % [3 × steps]

            acc_x = acc_x + m_b * abs(fftshift(fft(v_sum(1,:)))').^2;
            acc_y = acc_y + m_b * abs(fftshift(fft(v_sum(2,:)))').^2;
            acc_z = acc_z + m_b * abs(fftshift(fft(v_sum(3,:)))').^2;
        end

        SED_x(:,iq) = acc_x / norm;
        SED_y(:,iq) = acc_y / norm;
        SED_z(:,iq) = acc_z / norm;

        if mod(iq, 20) == 0 || iq == num_q
            elapsed = toc(t0);
            fprintf('  q %3d/%d  %.0fs  ETA %.0fs\n', ...
                iq, num_q, elapsed, elapsed/iq*(num_q-iq));
        end
    end

    freq_THz = linspace(-0.5/dt, 0.5/dt, steps_per_split) / 1e12;
    fprintf('[SED] Done in %.0fs.\n', toc(t0));
end


% ── 6. 시각화 ─────────────────────────────────────────────────────
function plot_SED(SED_x, SED_y, SED_z, freq_THz, q_reduced)
    fi    = freq_THz >= 0;
    qi    = q_reduced <= 0.5 + 1e-12;
    freq  = freq_THz(fi);
    q_plt = 2 * q_reduced(qi);

    S = SED_x(fi,qi) + SED_y(fi,qi) + SED_z(fi,qi);

%     % ── RGB (x→Blue, y→Green, z→Red) — 나중에 활성화 ──
%     clip = @(M) min(M, prctile(M(M>0), 99));
%     nx = clip(SED_x(fi,qi)); nx = nx/(max(nx(:))+eps);
%     ny = clip(SED_y(fi,qi)); ny = ny/(max(ny(:))+eps);
%     nz = clip(SED_z(fi,qi)); nz = nz/(max(nz(:))+eps);
%     xCol=[0 0 1]; yCol=[0 1 0]; zCol=[1 0 0]; N=2;
%     R=1-sqrt(((nx*(1-xCol(1))).^2+(ny*(1-yCol(1))).^2+(nz*(1-zCol(1))).^2)/N);
%     G=1-sqrt(((nx*(1-xCol(2))).^2+(ny*(1-yCol(2))).^2+(nz*(1-zCol(2))).^2)/N);
%     B=1-sqrt(((nx*(1-xCol(3))).^2+(ny*(1-yCol(3))).^2+(nz*(1-zCol(3))).^2)/N);
%     [xg,yg]=meshgrid(1:5,1:5); h=exp(-((xg-3).^2+(yg-3).^2)/2); h=h/sum(h(:));
%     RGB_s=cat(3,conv2(R,h,'same'),conv2(G,h,'same'),conv2(B,h,'same')).^0.9;

    fig = figure('Color','w','Position',[100 200 900 500]);
    imagesc(q_plt, freq, log(S + 1e-30));
    set(gca,'YDir','normal'); axis tight;
    ylim([0 1]);
    colormap(hot); colorbar;
    xlabel('q (π/a)'); ylabel('Frequency (THz)'); title('SED (log intensity)');

    saveName = fullfile(fileparts(mfilename('fullpath')), ...
        sprintf('SED_%s.png', datestr(now,'yyyymmdd_HHMMSS')));
    exportgraphics(fig, saveName, 'Resolution', 200);
    fprintf('Saved: %s\n', saveName);
end
