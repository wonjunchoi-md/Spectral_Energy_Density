%% Replot saved current-correlation data without recomputing.
clear; close all;
thisDir = fileparts(mfilename('fullpath'));
S = load(fullfile(thisDir, 'CC_data.mat'));
cfg = S.cfg;

% 0 means use the actual full min/max of log(C_L + C_T).
% After the first plot, set these to values such as -7 and -2.
cfg.dynasorLogMin = 0;
cfg.dynasorLogMax = 0;

if isfield(cfg, 'sourceDir') && exist(cfg.sourceDir, 'dir')
    addpath(cfg.sourceDir);
end

plot_current_process(S.CL, S.CT, S.omega_THz, S.q_reduced, cfg, thisDir, 'CC_process');
plot_current_saved(S.CL, S.CT, S.omega_THz, S.q_reduced, cfg, thisDir, 'CC_dynasor_replot');
