function [X, y] = read_data(filename)
% 	Read data from csvfile
% 
% 	Parameters:
% 	----------------------------------------------------
% 	filename: path to data file
% 
% 	Returns:
% 	----------------------------------------------------
% 	X: matrix of size (1, P)
% 	y: matrix of size (P, 1)
% 	
% 	Hints:
% 	----------------------------------------------------
% 	Make sure the sizes of X, y conform to above. You may
% 	find "csvread" useful

%% TODO

data = csvread(filename);
X_tmp = data(:,1);
y_tmp = data(:,2);

X = X_tmp';
y = y_tmp;

assert(size(X, 2) == size(data, 1), 'size of X incorrect');
assert(size(y, 1) == size(data, 1), 'size of y incorrect');