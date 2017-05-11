function F = poly_basis(X, D)
% 	Return polynomial basis for X (with ONE bias dimension)
% 
% 	Parameters:
% 	----------------------------------------------------
% 	X: data matrix of size (1, P)
% 	D: degree of Fourier basis features
% 
% 	Returns:
% 	----------------------------------------------------
% 	F: matrix of size (D+1, P)
% 	
% 	Hints:
% 	----------------------------------------------------
% 	Make sure the sizes of F conform to above. You may
% 	find ".^" useful

%% TODO

assert(size(F, 1)==D+1, 'degree incorrect');
assert(size(F, 2)==size(X, 2), 'number of data points incorrect');