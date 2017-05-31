% EECS 495: Homework 9
% Grad Student Extra Assignment
% 
% Modified by Stephanie Chang
% One vs. All multiclass classification

clear;
close all;
clc;

[X_train, y_train] = load_data('train_data_label.csv',1);
learning_rate = 0.001;
num_iter = 500;
W = train(X_train, y_train, num_iter, learning_rate);

y_pred = predict(W, X_train);
fprintf('The training accuracy is %f\n', mean(y_pred == y_train))

[X_test, y_test] = load_data('test_data.csv', 2);
y_pred = predict(W, X_test);
csvwrite('test_labels.csv', y_pred)

%% Auxilary Functions

function [X_data, y_data] = load_data(filename, set)
% Load the MNIST dataset
% X_data: N by 2680 matrix
% y_data: N by 1 matrix 
% N = 160 for train, N = 40 for test
%%
data = csvread(filename);
    if set == 1 %Train data set
        y_data = data(:, end);
        X_data = append_bias(data(:, 1:end-1));
    elseif set == 2 %Test data set
        y_data = zeros(size(data,1),1);
        X_data = append_bias(data(:, 1:end));
    end
end

function X_new = append_bias(X_old)
% Append an extra bias dimension to X_old as the FIRST column
% X_old: N by D matrix
% X_new: N by (D+1) matrix
% hint: you may find size() and ones() useful

%% TODO
new_col = ones(size(X_old,1),1);    %Nx1 vector of ones
X_new = cat(2, new_col, X_old);     %Tack on a column of ones to start of X_old 
end

function y_pred = predict(W, X)
% Calculate the scores in each class and predict the class label
% W: weight matrix, 2680 by 4
% X: N by 2680
% hint: you may find max(A, [], 2) very useful

%% TODO
A = X*W; %bj+Xp'W
[argmax, y_pred] = max(A, [], 2); %Extracting biggest value from each row
end

function W = train(X_train, y_train, num_iter, lr)
% Train the One-versus-All classifier
% X_train: N by 2680 matrix
% y_train: N by 1 matrix
% num_iter: number of iterations
% lr: learning rate
% W: 2680 by 4 matrix

D = size(X_train, 2); % 2680 Number of features
C = 4; % Number of classes
W =  zeros(D, C);

y_converted = convert_to_binary_class(y_train, C);
for i = 1: num_iter     % Loop 500 times
    grad = gradient_descent(W, X_train, y_converted);
    W = W - lr * grad;
end


end

function y_out = convert_to_binary_class(y_in, C)
% Convert y from a multiclass problem to a binary class one
% y_in: N by 1 matrix
% C: Number of class
% y_out: N by 4 matrix, each column consists of +1 or -1
% hint: you can use for loop for this part. And you may need logical indexing
%% TODO
n = size(y_in,1);   
y_out = -ones(n,C); %N x 4 
for row = 1:1:n     %Looking at each element in y_in
    for i = 1:1:C   %Checking classes 1-4
        if y_in(row) == i
            y_out(row,i) = 1;        
        end
    end
end

end

function grad = gradient_descent(W, X, y)
% Gradient descent for c-th classifier
% way to computing gradient on Canvas
%
% W: 2680 by 4 matrix
% X: N by 2680 matrix, already in compact notation
% y: N by 4 matrix, where each row contains labels per class
% grad: gradient of W, 2680 by 4 matrix
% hint: you may find sigmoid() below useful
%% TODO
    %%% initialize w0 and make step length %%%
    X = X';            % Flipping X = 2680 x N matrix with ones as top row
    
    % compute gradient of softmax      
    sig_in = -(X'*W).*(y);      % Nx4 matrix of -ypXp'w for all 4 classes
    r = -sigmoid(sig_in).*y;    % Nx4
    grad = X*r;                 % 2680x4
        

end


function y = sigmoid(z)
% Sigmoid function
y = zeros(size(z));
mask = (z >= 0.0);
y(mask) = 1./(1 + exp(-z(mask)));
y(~mask) = exp(z(~mask))./(1 + exp(z(~mask)));
end