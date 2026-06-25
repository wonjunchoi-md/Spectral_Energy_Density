function saveName = plot_current_process(CL, CT, omega_THz, q_reduced, cfg, outputDir, namePrefix)
% Show how the current-correlation plot changes during display processing.
    if nargin < 6 || isempty(outputDir)
        outputDir = fileparts(mfilename('fullpath'));
        if isempty(outputDir), outputDir = pwd; end
    end
    if nargin < 7 || isempty(namePrefix)
        namePrefix = 'CC_process';
    end
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    if ~isfield(cfg, 'dynasorLogMin'), cfg.dynasorLogMin = 0; end
    if ~isfield(cfg, 'dynasorLogMax'), cfg.dynasorLogMax = 0; end

    qi    = q_reduced <= 0.5 + 1e-12;
    q_plt = 2 * q_reduced(qi);

    Craw = max(CL(:, qi), 0) + max(CT(:, qi), 0);
    Craw(~isfinite(Craw)) = 0;

    CrawLog = log_with_floor(Craw);
    Csmooth = smooth_heatmap(Craw, 0.85, 0.50);
    CsmoothLog = log_with_floor(Csmooth);

    rawRange = finite_range(Craw);
    rawLogRange = finite_range(CrawLog);
    smoothLogRange = finite_range(CsmoothLog);
    [vmin, vmax] = resolve_log_limits(cfg, smoothLogRange(1), smoothLogRange(2));

    fig = figure('Color','w','Position',[80 100 1500 760]);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile(tl, 1);
    plot_one_panel(ax1, q_plt, omega_THz, Craw, rawRange, '1 raw linear');
    ylabel(ax1, 'Frequency (THz)');

    ax2 = nexttile(tl, 2);
    plot_one_panel(ax2, q_plt, omega_THz, CrawLog, rawLogRange, '2 raw log');

    ax3 = nexttile(tl, 3);
    plot_one_panel(ax3, q_plt, omega_THz, CsmoothLog, smoothLogRange, '3 smoothed log');
    ylabel(ax3, 'Frequency (THz)');

    ax4 = nexttile(tl, 4);
    plot_one_panel(ax4, q_plt, omega_THz, CsmoothLog, [vmin vmax], '4 selected log range');

    colormap(fig, inferno_colormap(256));

    cb = colorbar(ax4);
    cb.Layout.Tile = 'east';
    cb.Label.String = 'display intensity';
    cb.Label.FontSize = 12;

    tag = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    saveName = fullfile(outputDir, sprintf('%s_%s.png', namePrefix, tag));
    exportgraphics(fig, saveName, 'Resolution', 200);

    fprintf('Saved process view: %s\n', saveName);
    fprintf('raw linear range      : [%.6g, %.6g]\n', rawRange(1), rawRange(2));
    fprintf('raw log range         : [%.6g, %.6g]\n', rawLogRange(1), rawLogRange(2));
    fprintf('smoothed log range    : [%.6g, %.6g]\n', smoothLogRange(1), smoothLogRange(2));
    fprintf('selected log range    : [%.6g, %.6g]\n', vmin, vmax);
end

function plot_one_panel(ax, q_plt, omega_THz, Z, rangeVals, panelTitle)
    pcolormesh_centers(ax, q_plt, omega_THz(:), Z);
    set(ax, 'YDir', 'normal', 'Box', 'on', 'Layer', 'top', ...
        'FontSize', 12, 'LineWidth', 1.0, 'TickDir', 'in');
    hide_axes_toolbar(ax);
    axis(ax, 'tight');
    xlim(ax, [min(q_plt) max(q_plt)]);
    ylim(ax, [0 max(omega_THz)]);
    xticks(ax, 0:0.2:1);
    clim(ax, rangeVals);
    xlabel(ax, 'q (\pi/a)');
    title(ax, panelTitle);
end

function hide_axes_toolbar(ax)
    try
        if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
            ax.Toolbar.Visible = 'off';
        end
    catch
    end
end

function Zlog = log_with_floor(Z)
    vals = Z(isfinite(Z) & Z > 0);
    if isempty(vals)
        Zlog = zeros(size(Z));
        return;
    end
    floor_i = max(prctile(vals, 0.5), realmin);
    Zlog = log(Z + floor_i);
end

function rangeVals = finite_range(Z)
    vals = Z(isfinite(Z));
    if isempty(vals)
        rangeVals = [0 1];
        return;
    end
    rangeVals = [min(vals), max(vals)];
    if ~isfinite(rangeVals(1)) || ~isfinite(rangeVals(2)) || rangeVals(1) >= rangeVals(2)
        rangeVals = [0 1];
    end
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
        vmin = autoMin;
        vmax = autoMax;
    end
end

function Zs = smooth_heatmap(Z, sigma_y, sigma_x)
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
