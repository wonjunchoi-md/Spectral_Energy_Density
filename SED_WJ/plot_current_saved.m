function saveName = plot_current_saved(CL, CT, omega_THz, q_reduced, cfg, outputDir, namePrefix)
% Plot a saved current-correlation result without recomputing CL/CT.
    if nargin < 6 || isempty(outputDir)
        outputDir = fileparts(mfilename('fullpath'));
        if isempty(outputDir), outputDir = pwd; end
    end
    if nargin < 7 || isempty(namePrefix)
        namePrefix = 'CC_dynasor_replot';
    end
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    if ~isfield(cfg, 'dynasorLogMin'), cfg.dynasorLogMin = 0; end
    if ~isfield(cfg, 'dynasorLogMax'), cfg.dynasorLogMax = 0; end

    qi    = q_reduced <= 0.5 + 1e-12;
    q_plt = 2 * q_reduced(qi);

    C = max(CL(:, qi), 0) + max(CT(:, qi), 0);
    C(~isfinite(C)) = 0;
    C = smooth_heatmap(C, 0.85, 0.50);

    vals = C(isfinite(C) & C > 0);
    if isempty(vals)
        Clog = zeros(size(C));
        autoMin = 0;
        autoMax = 1;
    else
        floor_i = max(prctile(vals, 0.5), realmin);
        Clog = log(C + floor_i);
        finiteLog = Clog(isfinite(Clog));
        autoMin = min(finiteLog);
        autoMax = max(finiteLog);
    end

    [vmin, vmax] = resolve_log_limits(cfg, autoMin, autoMax);

    fig = figure('Color','w','Position',[120 120 820 520]);
    ax = axes(fig);
    pcolormesh_centers(ax, q_plt, omega_THz(:), Clog);
    set(ax, 'YDir', 'normal', 'Box', 'on', 'Layer', 'top', ...
        'FontSize', 13, 'LineWidth', 1.0, 'TickDir', 'in');
    axis(ax, 'tight');
    xlim(ax, [min(q_plt) max(q_plt)]);
    ylim(ax, [0 max(omega_THz)]);
    xticks(ax, 0:0.2:1);
    clim(ax, [vmin vmax]);
    colormap(ax, inferno_colormap(256));
    cb = colorbar(ax);
    cb.Label.String = 'log(C_L + C_T)';
    cb.Label.FontSize = 12;
    xlabel(ax, 'q (\pi/a)');
    ylabel(ax, 'Frequency (THz)');
    title(ax, 'Current spectrum');

    tag = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    saveName = fullfile(outputDir, sprintf('%s_%s.png', namePrefix, tag));
    exportgraphics(fig, saveName, 'Resolution', 200);

    fprintf('Saved: %s\n', saveName);
    fprintf('log color range used: [%.6g, %.6g]\n', vmin, vmax);
    fprintf('auto full log range : [%.6g, %.6g]\n', autoMin, autoMax);
end

function [vmin, vmax] = resolve_log_limits(cfg, autoMin, autoMax)
    vmin = cfg.dynasorLogMin;
    vmax = cfg.dynasorLogMax;

    if isempty(vmin) || isequal(vmin, 0)
        vmin = autoMin;
    end
    if isempty(vmax) || isequal(vmax, 0)
        vmax = autoMax;
    end

    if ~isfinite(vmin) || ~isfinite(vmax) || vmin >= vmax
        warning('Invalid dynasorLogMin/Max. Falling back to full auto range.');
        vmin = autoMin;
        vmax = autoMax;
    end

    if ~isfinite(vmin) || ~isfinite(vmax) || vmin >= vmax
        vmin = 0;
        vmax = 1;
    end
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

function cmap = inferno_colormap(n)
% Compact approximation of matplotlib's inferno colormap.
    if nargin < 1, n = 256; end
    x = linspace(0, 1, 10);
    c = [0.0015 0.0005 0.0139
         0.0874 0.0446 0.2248
         0.2582 0.0386 0.4065
         0.4163 0.0902 0.4329
         0.5783 0.1480 0.4044
         0.7357 0.2159 0.3302
         0.8650 0.3168 0.2261
         0.9545 0.4687 0.0999
         0.9876 0.6453 0.0399
         0.9871 0.9914 0.7495];
    xi = linspace(0, 1, n);
    cmap = interp1(x, c, xi, 'pchip');
    cmap = min(max(cmap, 0), 1);
end
