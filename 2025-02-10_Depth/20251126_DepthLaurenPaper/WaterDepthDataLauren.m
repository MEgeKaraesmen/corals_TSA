clc;
clear;
close all; % Closes previous figures and clears variables
a =12;
% Define depths and δ13C values
depths = [... % fill with actual data from output if needed
105*ones(1,9), ...
95*ones(1,9), ...
73*ones(1,10), ...
63*ones(1,11), ...
53*ones(1,10), ...
42*ones(1,10), ...
32.5*ones(1,10)];
depths = -depths -60;
d13c_values = [... % same length as depths
-2.8, -3.1, -1.5, -2.3, -2.7, -3.3, -3.0, -2.8, -3.0, ...
-1.1, -1.3, -1.8, -2.1, -2.6, -2.6, -2.2, -2.4, -1.8, ...
-2.9, -2.2, -2.2, -2.0, -2.2, -2.3, -2.2, -2.2, -1.9, -2.0, ...
-2.1, -2.1, -2.0, -2.0, -1.9, -1.8, -1.8, -2.3, -2.2, -2.4, -2.1, ...
-2.1, -2.3, -1.7, -2.4, -2.3, -2.0, -1.8, -1.9, -2.7, -3.1, ...
-1.4, -1.3, -1.7, -1.6, -1.4, -1.6, -1.7, -2.1, -2.5, -2.7, ...
-1.2, -1.1, -1.5, -1.5, -1.6, -1.7, -1.2, -1.4, -1.4, -1.6];

% Convert to log scale for fitting
y = depths(:);
d13C = d13c_values(:);
X = [ones(size(d13C)) d13C];

% Robust fit using iteratively reweighted least squares
[beta, stats] = robustfit(d13C,y);

% Generate prediction
d13C_range = linspace(min(d13C), max(d13C), 200)';
range = d13C_range;
X_pred = [ones(size(range)) range];
y_pred = X_pred * beta;

% Plotting
figure;
scatter( d13C,depths, 30, 'b', 'filled');
hold on;
plot(d13C_range, y_pred, 'r-', 'LineWidth', 2);
ylabel('Water Depth (cm)');
xlabel('\delta^{13}C');
%title('Robust Linear Fit: \delta^{13}C vs Water Depth');
legend('Data', 'Robust linear fit');
grid on;
set(gca, 'FontSize', 14);  %
% After plotting:

% Compute predicted values on original data
y_fit = X * beta;

% R^2
SS_res = sum((y - y_fit).^2);
SS_tot = sum((y - mean(y)).^2);
R2 = 1 - SS_res / SS_tot;
% Residuals
residuals = y - y_fit;

% Mean Square Error (MSE)
MS = mean(residuals.^2);

% Root Mean Square Error (RMSE) if you prefer in same units as depth
RMSE = sqrt(MS);

% Display
fprintf('MS (MSE) = %.4f\n', MS);
fprintf('RMSE = %.4f\n', RMSE);
% Annotate
annotation_text = sprintf('y = %.3f (x) + %.3f\nRMSE = %.3f', beta(2), beta(1), RMSE);
text(max(d13C) -1,min(depths) +20,  annotation_text, 'FontSize', 10, ...
     'BackgroundColor', 'white', 'EdgeColor', 'black', 'Margin', 5);
