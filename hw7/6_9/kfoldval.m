% EECS 495: Homework 7
% Problem 6.9
%
% Written by Stephanie Chang
% Reused bits of code from previous homeworks
%----------------------------------------
clear;
close all;
clc;

[X,y] = read_data('2eggs_data.csv');
num_fold = 3; num_degree = 8;
[fold_tr, fold_val] = random_split(length(y), num_fold); %Uneven splitting

MSE_train = zeros(1, num_degree); MSE_val = zeros(1, num_degree);
MSE_min = 1000;

for f = 1: num_fold
    [X_train, y_train, X_val, y_val] = split_data(X, y, fold_tr, fold_val);
    for d = 1: num_degree
        F_train = poly_basis(X_train, d)
        F_val = poly_basis(X_val, d);
        w = softmax_gradient_descent(F_train, y_train);
        MSE_train(d) = MSE_train(d) + mean_square_error(w, F_train, y_train)/num_fold;
        MSE_val(d) = MSE_val(d) + mean_square_error(w, F_val, y_val)/num_fold;
        if MSE_val(d) < MSE_min
            MSE_min = MSE_val(d);
            w_best = w;
        end
    end
end

[~, best_degree] = min(MSE_val);
fprintf('The best degree of polynomial basis, in terms of validation error, is %d\n',  best_degree);
make_plot([1: num_degree], MSE_train, MSE_val);
plot_data(X_train, y_train, X_val, y_val, best_degree, w_best)

function[X, y] = read_data(filename) 
% Getting X,y val from csv file

data = csvread(filename);
X_tmp = data(:,1);
y_tmp = data(:,2);

X = X_tmp'; % 1xP, P = 100
y = y_tmp;  % Px1

assert(size(X, 2) == size(data, 1), 'size of X incorrect');
assert(size(y, 1) == size(data, 1), 'size of y incorrect');
end

function [fold_tr, fold_val] = random_split(P, K) 
% Parsing points into 3 uneven groups

values = randperm(P,P);
fold_tr = reshape(values(1:2*(P-1)/K),[K-1,(P-1)/K]); %Training folds have 33 #s
fold_val = reshape(values((2*(P-1)/K)+1:P),[1,(P-1)/K + 1]); %Validation fold has 34 #s
end

function [X_train, y_train, X_val, y_val] = split_data(X, y, fold_tr, fold_val)
% Getting actual training and validation values using fold indices
K_tr = size(fold_tr, 1); %2
N_tr = size(fold_tr, 2); %33
K_val = size(fold_val, 1); %1
N_val = size(fold_val,2); %34

X_tmp = ones(K_tr, N_tr);
y_tmp = ones(N_tr, K_tr);
X_tmp_val = ones(K_val, N_val);
y_tmp_val = ones(N_val ,K_val);

% Making Xtrain and ytrain
for i = 1:K_tr
tr_ind = fold_tr(i,:);
    for k = 1:N_tr
        X_tmp(i,k) = X(1,tr_ind(k)); %2x33
        y_tmp(k,i) = y(tr_ind(k),1); %33x2
    end
end

X_train = cat(2, X_tmp(1,:), X_tmp(2,:)); %1x66
y_train = cat(1, y_tmp(:,1), y_tmp(:,2)); %66x1

% Making Xval and yval
for j = 1:K_val
test_ind = fold_val(j,:);
    for l = 1:N_val
        X_tmp_val(j,l) = X(1,test_ind(l)); %1x34
        y_tmp_val(l,j) = y(test_ind(l),1); %34x1
    end
end

X_val = X_tmp_val;
y_val = y_tmp_val;

end

%Xtrain = 2x33
function F = poly_basis(X, D) %Returns polynomial function
x_expand = ones(D+1,1)*X; % (D+1)xP, x1s all in first col
F = x_expand.^(linspace(0,D,D+1)'); 

assert(size(F, 1)==D+1, 'degree incorrect');
assert(size(F, 2)==size(X, 2), 'number of data points incorrect');
end

function w = softmax_grad_descent(F,y)
% Returns learned weights

    w = randn(3,1);     %initial guess
    alpha = 10^-2;
    
    %Set up
    iter = 1;
    max_its = 30000;
    grad = 1;
    
    N = size(F,1);  %3
    P = size(F,2);  %100
    
    while  norm(grad) > 10^-12 && iter < max_its
    % compute gradient
    sig_pow = (X'*w).*(y); % 100x1 matrix of ypXp'w
    sigma = 1./(ones(P,1)+exp(sig_pow));
    r = -(sigma).*y; %100x1
    grad = X*r;           % YOUR CODE GOES HERE
    w = w - alpha*grad;
    % update iteration count
    iter = iter + 1;
    end
    
end

function mse = mean_square_error(w,F,y)
e = (F'*w - y).^2;
mse = mean(e);
end

function make_plot(D, MSE_train, MSE_val)
figure(1);
hold on;
plot(D, MSE_train, 'yv--');
plot(D, MSE_val, 'bv--');
legend('training error', 'validation error', 'Location', 'northwest');
ax = gca;
ax.YScale = 'log';
xlabel('Degree of basis');
ylabel('Error in log scale');
end

function plot_data(X_train,y_train,X_val,y_val, best_degree, w_best)
    %Plotting data points
    figure(2)
    hold on
    scatter(X_train,y_train,'fill');
    scatter(X_val,y_val,'fill');
    str = sprintf('Best Fit Model using Polynomial Basis, D = %d', best_degree);
    title(str);
    xlabel('x');
    ylabel('y');
    
    %Plotting best model
    xstore = [];
    fstore = [];
    
    for x = 0:1:10
        xlist = (x*ones(1,best_degree)).^linspace(1,best_degree,best_degree);
        f = w_best(1,1)+xlist*w_best(2:end,1);  
        fstore = cat(2,fstore,f);
        xstore = cat(2,xstore,x);
    end
    plot(xstore, fstore);
    
end
