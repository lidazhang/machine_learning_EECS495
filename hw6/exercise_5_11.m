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
MSE_min = 1000;

for d = 1: num_degree
    F_train = fourier_basis(X_train, d);
    F_val = fourier_basis(X_val, d);
    w = least_square_sol(F_train, y_train);
    MSE_train(d) = mean_square_error(w, F_train, y_train);
    MSE_val(d) = mean_square_error(w, F_val, y_val);
    if MSE_val(d) < MSE_min
        MSE_min = MSE_val(d);
        w_best = w;
    end
end

[~, best_degree] = min(MSE_val);
fprintf('The best degree of Fourier basis, in terms of validation error, is %d\n',  best_degree);
make_plot([1: num_degree], MSE_train, MSE_val);
plot_data(X_train,y_train,X_val,y_val,best_degree,w_best);

function plot_data(X_train,y_train,X_val,y_val, best_degree, w_best)
    %Plotting data points
    figure(2)
    hold on
    scatter(X_train,y_train,'fill');
    scatter(X_val,y_val,'fill');
    str = sprintf('Best Fit Model using Fourier Basis, D = %d', best_degree);
    title(str);
    xlabel('x');
    ylabel('y');
    
    %Plotting best model
    xstore = [];
    fstore = [];

    wx2 = reshape(w_best(2:end,:),2,[])';  %W = [w2,w3; w4,w5; ...]
    for x = 0:0.01:1
        col = 2*pi*x*ones(size(wx2,1),1).*linspace(1,best_degree,best_degree)';
        trig = [cos(col), sin(col)];
        f = w_best(1,1)+sum(sum(trig.*wx2,2),1);  
        fstore = cat(2,fstore,f);
        xstore = cat(2,xstore,x);
    end
    plot(xstore, fstore);
end


