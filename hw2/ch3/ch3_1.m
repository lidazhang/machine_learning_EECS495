function ch3_1()
% Problem 3.1
% Written by Stephanie Chang
%
% Linear model for student_debt.csv
%-------------------------------------------------------------------
% Importing values from .csv
data = csvread('student_debt.csv');
year = data(:,1); %40x1, Select element using year(#)
debt = data(:,2); %40x1, Select element using debt(#)

% Finding w*~ using 3.12
x11 = size(data,1); % x11 = P*1 = 40
xdiag = sum(year, 1); % x12 = x21 = x1 +...+ xP 
x22 = sum(year.^2,1); % x22 = x1^2 +...+ xP^2
outer_product = [x11 xdiag; xdiag x22];
wstartilde = pinv(outer_product)*[sum(debt); sum(year.*debt)]; % 2x1

% Parsing w*~ to get w* and b*
bstar = wstartilde(1);
wstar = wstartilde(2);

% Predict debt in 2050 using 3.14 
x_new = 2050;
debt_new = bstar + x_new'*wstar

% Plotting linear model and actual data  
debt_fit = bstar*ones([40, 1]) + wstar.*year;
fit_line = plot(year, debt_fit, 'r')
hold on

given_data = scatter(year, debt, 'filled', 'b')
hold on
title('Student Debt vs. Year')
xlabel('Year')
ylabel('Student Debt (Trillions)')
end

