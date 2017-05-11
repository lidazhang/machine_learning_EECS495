clear;
close all;
clc;
%------------------------------------------------------------------------
% EECS 495: Homework 6
% Modified by Stephanie Chang
% Problem 5.11
%------------------------------------------------------------------------

[X, y] = read_data('wavy_data.csv');
num_fold = 3; num_degree = 8;
folds = random_split(length(y), num_fold);
[X_train, y_train, X_val, y_val] = train_val_split(X, y, folds, 1);

MSE_train = zeros(1, num_degree); MSE_val = zeros(1, num_degree);

for d = 1: num_degree
    F_train = fourier_basis(X_train, d);
    F_val = fourier_basis(X_val, d);
    w = least_square_sol(F_train, y_train);
    MSE_train(d) = mean_square_error(w, F_train, y_train);
    MSE_val(d) = mean_square_error(w, F_val, y_val);
end

[~, best_degree] = min(MSE_val);
fprintf('The best degree of Fourier basis, in terms of validation error, is %d\n',  best_degree);
make_plot([1: num_degree], MSE_train, MSE_val);

