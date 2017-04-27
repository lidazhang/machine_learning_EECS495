function [X_data, y_data] = load_MNIST(filename)
% Load the MNIST dataset
% X_data: N by 785 matrix
% y_data: N by 1 matrix
data = csvread(filename);
y_data = data(:, end);
X_data = append_bias(data(:, 1:end-1));
end

function X_new = append_bias(X_old)
% Append an extra bias dimension to X_old as the FIRST column
% X_old: N by D matrix
% X_new: N by (D+1) matrix
% hint: you may find size() and ones() useful

%% TODO
new_col = ones(size(X_old,2),1);    %Nx1 vector of ones
X_new = cat(2, X_old, new_col);     %Tack on a column of ones to X_old 
end