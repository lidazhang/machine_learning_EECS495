clear;
close all;
clc;

[X, y] = read_data('galileo_ramp_data.csv');
num_fold = 6; num_degree = 6;
folds = random_split(length(y), num_fold);

MSE_train = zeros(1, num_degree); MSE_val = zeros(1, num_degree);

for f = 1: num_fold
    [X_train, y_train, X_val, y_val] = train_val_split(X, y, folds, f);
    for d = 1: num_degree
        F_train = poly_basis(X_train, d);
        F_val = poly_basis(X_val, d);
        w = least_square_sol(F_train, y_train);
        MSE_train(d) = MSE_train(d) + mean_square_error(w, F_train, y_train)/num_fold;
        MSE_val(d) = MSE_val(d) + mean_square_error(w, F_val, y_val)/num_fold;        
    end
end

[~, best_degree] = min(MSE_val);
fprintf('The best degree of polynomial basis, in terms of validation error, is %d\n',  best_degree);
make_plot([1: num_degree], MSE_train, MSE_val);