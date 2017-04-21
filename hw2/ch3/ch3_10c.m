function ch3_10c()
% Problem 3.10c
% Written by Stephanie Chang
%
% Solving the Least Squares cost function 
% Plotting the data with the logistic sigmoid fit
%------------------------------------------------------------------------
% Importing values from .csv
bact_data = csvread('bacteria_data.csv');
hour = bact_data(:,1);
growth = bact_data(:,2)

% Finding sigmoid points

% Outer product term
x11 = size(bact_data,1); % x11 = P*1 = 40
xdiag = sum(hour, 1); % x12 = x21 = x1 +...+ xP 
x22 = sum(hour.^2,1); % x22 = x1^2 +...+ xP^2
i_outer_product = pinv([x11 xdiag; xdiag x22]);

% xp~ log term
ln_input = growth./(ones([9,1])-growth);
ln_part = log(ln_input);

% Finding w* and b*
wstar_tilde = i_outer_product*[sum(ln_part); sum(hour.*ln_part)];
bstar = wstar_tilde(1);
wstar = wstar_tilde(2);

% Finding and plotting the sigmoid curve
input = bstar*ones([9,1]) + hour.*wstar
pt1 = 1/(1+exp(1)^(-input(1)))
pt2 = 1/(1+exp(1)^(-input(2)))
pt3 = 1/(1+exp(1)^(-input(3)))
pt4 = 1/(1+exp(1)^(-input(4)))
pt5 = 1/(1+exp(1)^(-input(5)))
pt6 = 1/(1+exp(1)^(-input(6)))
pt7 = 1/(1+exp(1)^(-input(7)))
pt8 = 1/(1+exp(1)^(-input(8)))
pt9 = 1/(1+exp(1)^(-input(9)))
pt = [pt1; pt2; pt3; pt4; pt5; pt6; pt7; pt8; pt9]
sigma = plot(hour, pt)
hold on

% Plotting the data with the sigmoid fit
population_change = scatter(hour,growth, 'filled')
hold on
title('Normalized Bacteria Cell Concentration vs. Hours')
xlabel('Hour')
ylabel('Normalized Cell Concentration')
end

