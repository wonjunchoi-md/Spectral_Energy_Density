%% ================================================================
%  Current Correlation Function — C_L(q,ω), C_T(q,ω)
%
%  j(q,t)   = Σ_i exp(iq·r_i(t)) · v_i(t)   [actual positions]
%  j_L(q,t) = j · q̂                           [longitudinal]
%  j_T(q,t) = j - j_L·q̂                       [transverse]
%
%  C_L(q,ω) = |FFT[j_L(q,t)]|² / (N · dt)
%  C_T(q,ω) = 0.5 · Σ_α |FFT[j_T_α(q,t)]|² / (N · dt)
% ================================================================

%% ===== Config ===================================================
cfg.folderPath = '';               % 비워두면 GUI 폴더 선택창
cfg.primFile   = 'prim_no_H.xyz';  % 선택한 폴더 안에 있어야 함
cfg.timeStepFs = 40;
cfg.maxSteps   = 0;

%% ===== Pipeline =================================================

[atoms, mData, folderPath] = read_trajectory(cfg.folderPath, cfg.maxSteps);
prim           = read_prim_xyz(fullfile(folderPath, cfg.primFile));
ref            = build_reference(prim, mData, atoms);

% BvK q-points (동일한 방식)
q_reduced = (0 : floor(ref.N_UC/2)) / ref.N_UC;
q_cart    = make_q_path(ref, q_reduced);

[freq_THz, CL, CT] = compute_current_correlate(atoms, q_cart, cfg);
plot_CC(CL, CT, freq_THz, q_reduced);


%% ================================================================
%  Local Functions
%% ================================================================

function [freq_THz, CL, CT] = compute_current_correlate(atoms, q_cart, cfg)
% C_L(q,ω), C_T(q,ω) via FFT of current j(q,t)

    Nt   = size(atoms.vel, 3);
    dt   = cfg.timeStepFs * 1e-15;
    num_q = size(q_cart, 1);

    % q̂ (단위벡터) [Nq × 3]
    q_mag = sqrt(sum(q_cart.^2, 2));
    q_hat = q_cart ./ (q_mag + eps);

    fprintf('[CC] %d q-pts × %d steps ...\n', num_q, Nt);
    t0 = tic;

    % j_L, j_T 시계열 누적 [Nq × Nt]
    jL = zeros(num_q, Nt);           % complex
    jT = zeros(num_q, 3, Nt);        % complex

    for it = 1:Nt
        r     = atoms.pos(:, 2:4, it);        % [N × 3]  실제 위치
        v     = atoms.vel(:, 2:4, it);        % [N × 3]
        phase = exp(1i * (r * q_cart'));       % [N × Nq]
        j     = phase' * v;                    % [Nq × 3]

        jL_t       = sum(j .* q_hat, 2);      % [Nq × 1]
        jL(:,it)   = jL_t;
        jT(:,:,it) = j - jL_t .* q_hat;       % [Nq × 3]

        if mod(it, 1000) == 0
            fprintf('  step %d/%d  (%.0fs)\n', it, Nt, toc(t0));
        end
    end

    % FFT → PSD
    norm  = Nt * dt;
    CL    = zeros(Nt, num_q);
    CT    = zeros(Nt, num_q);

    for iq = 1:num_q
        FL        = fftshift(fft(jL(iq,:)));
        CL(:,iq)  = abs(FL).^2 / norm;

        FTx = fftshift(fft(squeeze(jT(iq,1,:))'));
        FTy = fftshift(fft(squeeze(jT(iq,2,:))'));
        FTz = fftshift(fft(squeeze(jT(iq,3,:))'));
        CT(:,iq)  = 0.5 * (abs(FTx).^2 + abs(FTy).^2 + abs(FTz).^2) / norm;
    end

    freq_THz = linspace(-0.5/dt, 0.5/dt, Nt) / 1e12;
    fprintf('[CC] Done in %.0fs.\n', toc(t0));
end


function plot_CC(CL, CT, freq_THz, q_reduced)
    fi    = freq_THz >= 0;
    qi    = q_reduced <= 0.5 + 1e-12;
    freq  = freq_THz(fi);
    q_plt = 2 * q_reduced(qi);

    figure('Color','w','Position',[100 100 1200 500]);

    subplot(1,2,1);
    imagesc(q_plt, freq, log(CL(fi,qi) + 1e-30));
    set(gca,'YDir','normal'); axis tight; ylim([0 1]);
    colormap(hot); colorbar;
    xlabel('q (π/a)'); ylabel('Frequency (THz)'); title('C_L (longitudinal)');

    subplot(1,2,2);
    imagesc(q_plt, freq, log(CT(fi,qi) + 1e-30));
    set(gca,'YDir','normal'); axis tight; ylim([0 1]);
    colormap(hot); colorbar;
    xlabel('q (π/a)'); ylabel('Frequency (THz)'); title('C_T (transverse)');

    saveName = fullfile(fileparts(mfilename('fullpath')), ...
        sprintf('CC_%s.png', datestr(now,'yyyymmdd_HHMMSS')));
    exportgraphics(gcf, saveName, 'Resolution', 200);
    fprintf('Saved: %s\n', saveName);
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
