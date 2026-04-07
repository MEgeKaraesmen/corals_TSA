clear
clc


% velocities from the growing window stack
velocities = [26.4, 19.5, 14.2, 12.7, 10.5, 6.2, 5.9, 5.9];

% SAr acquisitons
SAR_acq_dates = [decyear(2007, 04, 16), decyear(2007, 09, 03), decyear(2007, 10, 17), decyear(2008, 01, 17), decyear(2008, 03, 03), decyear(2008, 04, 18), decyear(2008, 09, 01), decyear(2009, 01, 19), decyear(2010, 01, 22)];

% days the growing windows cover
days = [138, 184, 276, 322, 368, 506, 644, 1012]; 

days_mid = days / 2;

% Load in the Coral Data
load("postseismic_raw_dates2.mat");
load("postseismic_raw_data2.mat");

%% Fit an exp curve to the velocity vs. middle points to get "middle" estimate at t7 and t8

tau0 = median(days_mid); % decay factor        
p0 = [velocities(1), median(days_mid)]; % initial guess for nonlinear fit

opts = statset('nlinfit');
opts.RobustWgtFun = 'bisquare';  % helps with noisy windows by downweighting outliers

model_exp = @(p,dt) p(1) .* (p(2)./dt) .* (1 - exp(-dt./p(2))); % defines our expected model (see handwritten notes)

[p,resid,J,CovB] = nlinfit(days_mid, velocities, model_exp, p0, opts);

v0    = p(1); % fitted v0 from our growing window velocity
tau   = p(2); % fitted tau from our growing window velocity   

dt_reg = linspace(min(days_mid), max(days_mid), 20); % 22 samples would give regular ALOS
v_fit_exp = model_exp(p, dt_reg);

dt_reg_ext = linspace(min(days_mid), max(days), 42); % 22 samples would give regular ALOS
v_fit_exp_ext   = model_exp(p, dt_reg_ext);

figure() 
plot(days_mid, velocities, 'ko', 'MarkerFaceColor','k')
hold on
plot(dt_reg_ext, v_fit_exp_ext, 'r', 'LineWidth',2)
plot(dt_reg, v_fit_exp, 'g', 'LineWidth',2)
xline(644)
xline(1012)
xlabel('Days Since First SAR Acq')
ylabel('Velocity')
title('Exponential Fit')
legend('Observed Window velocities','Exponential fit extended', 'Exponential fit')
grid on

% extract velocity at t = 14 (644 days) and t = 22 (1012 days)

v7_exp = v_fit_exp_ext(find(dt_reg_ext == 644));
v8_exp = v_fit_exp_ext(find(dt_reg_ext == 1012));


%% Fit a log curve to the velocity vs. middle points to get "middle" estimate at t7 and t8

opts = statset('nlinfit');
opts.RobustWgtFun = 'bisquare';  % helps with noisy windows by downweighting outliers

% logarithmic decay overlapping window velocity eq (see handwritten notes)
model_log = @(p,dt)(p(1)./dt) .* log((dt + p(2))./p(2));

% initial guesses for nonlinear fit
v0_0 = velocities(1) * days_mid(1);   % rough scale
c0  = min(days_mid)/2;        % small offset
p0 = [v0_0, c0];

[p,resid,J,CovB] = nlinfit(days_mid, velocities, model_log, p0, opts);

% fitted v0 and c from our velocity data
v0 = p(1);
c = p(2);

dt_reg = linspace(min(days_mid), max(days_mid), 20); % 22 samples would give regular ALOS
v_fit_log = model_log(p, dt_reg);

dt_reg_ext = linspace(min(days_mid), max(days), 42); % 22 samples would give regular ALOS
v_fit_log_ext   = model_log(p, dt_reg_ext);

figure()
hold on
plot(days_mid, velocities, 'ko', 'MarkerFaceColor','k')
hold on
plot(dt_reg_ext, v_fit_log_ext, 'r', 'LineWidth',2)
plot(dt_reg, v_fit_log, 'g', 'LineWidth',2)
xline(644)
xline(1012)
xlabel('Days Since First SAR Acq')
ylabel('Velocity')
title('Logarithmic Fit')
legend('Observed window velocities','Logarithmic fit extended', 'Logarithmic fit')
grid on

% extract velocity at t = 14 (644 days) and t = 22 (1012 days)

v7_log = v_fit_log_ext(find(dt_reg_ext == 644));
v8_log = v_fit_log_ext(find(dt_reg_ext == 1012));

%% Using Middle Dates in Incremental Analysis

% See handwritten notes
d0 = 0;
d1 = (velocities(1) * days_mid(1) + velocities(2) * (days_mid(2) - days_mid(1)) + velocities(2) * (days_mid(3) - days_mid(2)))/365; 
d2 = d1 + (velocities(4) * (days_mid(4) - days_mid(3)) + velocities(5) * (days_mid(5) - days_mid(4)))/365;

% need to interpolate a value to get us to t=6
% nonlinear interpolation between v6 and v7, days_mid(6:7)
% we have t_mid(6) = 5.5 and t_mid(7) = 7. Need a t_mid(int) = 6

tmid_int = linspace(days_mid(6), days_mid(7), 4);
vmid_int = interp1(days_mid(6:7),velocities(6:7),tmid_int,'spline');

d3 = d2 + (velocities(6) * (days_mid(6) - days_mid(5)) + vmid_int(2) * (tmid_int(2) - days_mid(6)))/365;
d4 = d3 + (velocities(7) * (days_mid(7)-tmid_int(2)))/365;

% need time value broken up between 7 and 11 (t_mid(8) and t_mid(7))
tmid_int2 = linspace(days_mid(7), days_mid(8), 5);
vmid_int2 = interp1(days_mid(7:8),velocities(7:8),tmid_int2,'spline');

d5 = d4 + (vmid_int2(2) * (tmid_int2(2)-days_mid(7)))/365;
d6 = d5 + (velocities(8) * (days_mid(end)-tmid_int2(2)))/365;

% USING EXPONENTIAL FIT
d7 = d6 + (v7_exp * (days(7)-days_mid(8))/365)
d8 = d7 + (v8_exp * (days(8)-days(7))/365)

def = [d0, d1, d2, d3, d4, d5, d6, d7, d8];

figure()
yyaxis left
plot(SAR_acq_dates, def, '-o','Color','b','MarkerFaceColor','b','LineWidth',4,'MarkerSize',10)
yticks(linspace(0,25,6))
ylim([0 25])
ax = gca; 
set(gca,'Ydir','reverse')
grid on
title('Exponential Fit for V7 and V8')

yyaxis right
plot(raw_dates(5:39), postEQ_coral(5:39),':*','LineWidth',2,'MarkerSize',6,'Color','r')
ylabel('\delta^{13}C RSL (cm)', 'Interpreter','tex');
yticks(linspace(-170,-80,4))
ax.FontSize = 18; 
ax.YAxis(2).FontWeight = 'bold';
ax.YAxis(2).Color = 'r'; % Sets the left y-axis color 
ylim([-170 -80])

% USING LOG FIT
d7 = d6 + (v7_log * (days(7)-days_mid(8))/365)
d8 = d7 + (v8_log * (days(8)-days(7))/365)

def = [d0, d1, d2, d3, d4, d5, d6, d7, d8];

figure()
yyaxis left
plot(SAR_acq_dates, def,'-o','Color','b','MarkerFaceColor','b','LineWidth',4,'MarkerSize',10)
yticks(linspace(0,25,6))
ylim([0 25])
ax = gca; 
set(gca,'Ydir','reverse')
grid on
title('Logarthmic Fit for V7 and V8')

yyaxis right
plot(raw_dates(5:39), postEQ_coral(5:39),':*','LineWidth',2,'MarkerSize',6,'Color','r')
ylabel('\delta^{13}C RSL (cm)', 'Interpreter','tex');
yticks(linspace(-170,-80,4))
ax.FontSize = 18; 
ax.YAxis(2).FontWeight = 'bold';
ax.YAxis(2).Color = 'r'; % Sets the left y-axis color 
ylim([-170 -80])








