function [X_train, y_train, X_val, y_val] = train_val_split(X, y, folds, fold_id)
%   Split the data into training and validation sets
% 
% 	Parameters:
% 	----------------------------------------------------
% 	X: matrix of size (1, N)
% 	y: matrix of size (N, 1)
% 	folds: a matrix of size (K, P/K), elements at k-th
% 	        correspond to position indices in k-th fold
% 	fold_id: the id of the fold you want to be validation set
% 
% 	Returns:
% 	----------------------------------------------------
% 	X_train: training set of X
% 	y_train: training label
% 	X_val: validation set of X
% 	y_val: validation label

%% TODO

assert(length([y_val; y_train]) == length(y), 'Split incorrect');