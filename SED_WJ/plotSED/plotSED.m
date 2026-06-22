function plot_SED(sed_x, sed_y, sed_z, freq_THz, q_reduced)
% ================================================================
%   HIGH-QUALITY SED VISUALIZATION
%   - SED_x  (White → Blue)
%   - SED_y  (White → Green)
%   - SED_z  (White → Red)
%   - Combined RGB (posvelClass 방식)
%
%   Author: ChatGPT
% ================================================================


%% =============================
%   0. RANGE CROPPING
% =============================
freq_idx = freq_THz >= 0;
q_idx    = (q_reduced >= 0) & (q_reduced <= 0.5 + 1e-12);

freq_THz   = freq_THz(freq_idx);
q_reduced  = q_reduced(q_idx);

sed_x = sed_x(freq_idx, q_idx);
sed_y = sed_y(freq_idx, q_idx);
sed_z = sed_z(freq_idx, q_idx);


%% =============================
%   1. BASE FIGURE FOR AX1–AX3
% =============================
figure('Color','w','Position',[200 200 1200 900]);


%% =============================
%   2. COLOR MAPS
% =============================
% White → Blue
cmap_blue  = [ linspace(1,0,256)' , linspace(1,0,256)' , ones(256,1) ];

% White → Green
cmap_green = [ linspace(1,0,256)' , ones(256,1) , linspace(1,0,256)' ];

% White → Red
cmap_red   = [ ones(256,1) , linspace(1,0,256)' , linspace(1,0,256)' ];


%% =============================
%   2-1. SED_x (White → Blue)
% =============================
ax1 = subplot(2,2,1);
imagesc(ax1, q_reduced, freq_THz, sed_x);
set(ax1,'YDir','normal');
xlabel(ax1,'q'); ylabel(ax1,'THz');
title(ax1,'SED_x (White → Blue)');
colorbar(ax1);
colormap(ax1, cmap_blue);


%% =============================
%   2-2. SED_y (White → Green)
% =============================
ax2 = subplot(2,2,2);
imagesc(ax2, q_reduced, freq_THz, sed_y);
set(ax2,'YDir','normal');
xlabel(ax2,'q'); ylabel(ax2,'THz');
title(ax2,'SED_y (White → Green)');
colorbar(ax2);
colormap(ax2, cmap_green);


%% =============================
%   2-3. SED_z 99% CLIPPED (White → Red)
% =============================
ax3 = subplot(2,2,3);

thr_z = prctile(sed_z(sed_z>0), 99);
sed_z_clip = min(sed_z, thr_z);

imagesc(ax3, q_reduced, freq_THz, sed_z_clip);
set(ax3,'YDir','normal');
xlabel(ax3,'q'); ylabel(ax3,'THz');
title(ax3,'SED_z (White → Red, 99% clip)');
colorbar(ax3);
colormap(ax3, cmap_red);



%% ===============================================================
%   3. COMBINED RGB (posvelClass 방식)
% ===============================================================

% ------- 3-1. clipping (intensity too large → fold down)
clipSED = @(M) min(M, prctile(M(M>0),99));

sed_x_c = clipSED(sed_x);
sed_y_c = clipSED(sed_y);
sed_z_c = clipSED(sed_z);

% ------- 3-2. normalization (0~1)
nx = sed_x_c / max(sed_x_c(:)+eps);
ny = sed_y_c / max(sed_y_c(:)+eps);
nz = sed_z_c / max(sed_z_c(:)+eps);

% ------- 3-3. weight (posvelClass와 구조 동일)
wx = 1; wy = 1; wz = 1;

% ------- 3-4. color basis (x→Blue, y→Green, z→Red)
xCol = [0 0 1];
yCol = [0 1 0];
zCol = [1 0 0];

avgNum = 2;    % posvelClass의 안정화 파라미터

% ------- 3-5. RGB CHANNEL BUILDING (posvelClass 핵심)
R = 1 - sqrt( ( (nx.*wx*(1-xCol(1))).^2 + ...
                (ny.*wy*(1-yCol(1))).^2 + ...
                (nz.*wz*(1-zCol(1))).^2 ) / avgNum );

G = 1 - sqrt( ( (nx.*wx*(1-xCol(2))).^2 + ...
                (ny.*wy*(1-yCol(2))).^2 + ...
                (nz.*wz*(1-zCol(2))).^2 ) / avgNum );

B = 1 - sqrt( ( (nx.*wx*(1-xCol(3))).^2 + ...
                (ny.*wy*(1-yCol(3))).^2 + ...
                (nz.*wz*(1-zCol(3))).^2 ) / avgNum );

% ------- 3-6. STACK RGB MAP
RGB_combined = cat(3, R, G, B);



%% =============================
%   4. Combined RGB Plot
% =============================
figure('Color','w','Position',[200 200 1000 750]);
axC = axes;

image(axC, 2*q_reduced, freq_THz, RGB_combined);
set(axC,'YDir','normal');
xlabel(axC,'Wavevector (pi/a)');
ylabel(axC,'THz');
title(axC,'SED (Combined RGB — posvelClass method)');



%% ===============================================================
%   5. EXTRA: SED_z LOG + RE-LU FILTERING
% ===============================================================
figure('Color','w','Position',[200 200 900 700]);
ax5 = axes;

L = log10(sed_z + eps) / log(400);
thr = prctile(L(:), 90);
L_relu = L;
L_relu(L < thr) = 0;

imagesc(ax5, q_reduced, freq_THz, L_relu);
set(ax5,'YDir','normal');
xlabel(ax5,'q'); ylabel(ax5,'THz');
title(ax5,'SED_z (log + ReLU 98%)');
colorbar(ax5);
colormap(ax5, cmap_red);

end