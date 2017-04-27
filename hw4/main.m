% EECS 495 Homework 4: Problem 4.16
% OvA Starter Code
% Modified by Stephanie Chang
%-------------------------------------------------------------------------
clear;
close all;
clc;

[X_train, y_train] = load_MNIST('MNIST_data/MNIST_train_data.csv');
learning_rate = 0.01;
num_iter = 1000;
W = train(X_train, y_train, num_iter, learning_rate); %WIP

y_pred = predict(W, X_train);   %WIP
fprintf('The training accuracy is %f\n', mean(y_pred == y_train))

[X_test, y_test] = load_MNIST('MNIST_data/MNIST_test_data.csv');
y_pred = predict(W, X_test);
fprintf('The test accuracy is %f\n', mean(y_pred == y_test))