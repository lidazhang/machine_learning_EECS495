function F = fourier_basis(X, D)
% 	Return Fourier basis for X (with ONE bias dimension)
% 
% 	Parameters:
% 	----------------------------------------------------
% 	X: data matrix of size (1, P)
% 	D: degree of Fourier basis features
% 
% 	Returns:
% 	----------------------------------------------------
% 	F: matrix of size (2D+1, P)
% 	
% 	Hints:
% 	----------------------------------------------------
% 	Make sure the sizes of F conform to above. You may
% 	find "size, reshape, ones, cos, sin" useful

%% TODO

assert(size(F, 1)==2*D+1, 'degree incorrect');
assert(size(F, 2)==size(X, 2), 'number of data points incorrect');