%% SeaLevelFromMorphology.m
clear; close all; clc;

fname = "SeaLevelFromMorphology.csv";
xerr  = 1;   % +/- 1 year (dating uncertainty)

T = readtable(fname, 'VariableNamingRule','preserve');

t_raw    = string(T{:,1});
spA_raw  = string(-(T{:,2} + 52.3581)) ;
splA_raw  = string(-(T{:,3}+107));

t    = str2double(regexprep(t_raw,   '[^0-9\.\-]+', ''));
spA  = str2double(regexprep(spA_raw , '[^0-9\.\-]+', ''));
splA = str2double(regexprep(splA_raw,'[^0-9\.\-]+', ''));

ok1 = isfinite(t) & isfinite(spA);
ok2 = isfinite(t) & isfinite(splA);

%% --- Figure ---
%% --- Plot ---
fig = figure('Color','w');
ax  = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
ax.FontSize = 14;

xerr = 1;   % +/- 1 year

% Data
t1 = t(ok1); y1 = spA(ok1);
t2 = t(ok2); y2 = splA(ok2);

% Choose colors (any scheme you like)
c1 = [0.00 0.45 0.74];   % blue  (12 SP-A)
c2 = [0.85 0.33 0.10];   % red   (12 SPL-A)

% ---- 12 SP-A ----
plot(ax, t1 - xerr, y1, '--', 'Color', c1, 'LineWidth', 1.0);
p1 = plot(ax, t1,        y1, '-',  'Color', c1, 'LineWidth', 2.2);
plot(ax, t1 + xerr, y1, '--', 'Color', c1, 'LineWidth', 1.0);
plot(ax, t1, y1, 'o', 'Color', c1, 'MarkerFaceColor', c1);

% ---- 12 SPL-A ----
plot(ax, t2 - xerr, y2, '--', 'Color', c2, 'LineWidth', 1.0);
p2 = plot(ax, t2,        y2, '-',  'Color', c2, 'LineWidth', 2.2);
plot(ax, t2 + xerr, y2, '--', 'Color', c2, 'LineWidth', 1.0);
plot(ax, t2, y2, 's', 'Color', c2, 'MarkerFaceColor', c2);

xlabel(ax, 'Year (CE)');
ylabel(ax, 'HLS water depth (cm)');
%title(ax, 'HLS water depth with dating uncertainty (±1 yr)');

legend(ax, [p1 p2], {'12 SP-A', '12 SPL-A'}, 'Location','best');


%% --- Robust axis limits ---
t_ok = t(isfinite(t));
if ~isempty(t_ok)
    xlim(ax, [min(t_ok)-2-xerr, max(t_ok)+2+xerr]);
end

y_ok = [spA(ok1); splA(ok2)];
y_ok = y_ok(isfinite(y_ok));
if ~isempty(y_ok)
    ylo = min(y_ok); yhi = max(y_ok);
    if yhi <= ylo, pad = 1; else, pad = 5; end
    ylim(ax, [ylo-pad, yhi+pad]);
end
