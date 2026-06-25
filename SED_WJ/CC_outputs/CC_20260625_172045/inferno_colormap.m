function cmap = inferno_colormap(n)
% Compact approximation of matplotlib's inferno colormap.
    if nargin < 1
        n = 256;
    end

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
