function w = least_square_sol(F, y)
% 	Refer to eq. 5.19 in the text
% 
% 	Parameters:
% 	----------------------------------------------------
% 	F: matrix of shape (2D+1 or D+1 depends on what basis, P)
% 	y: matrix of shape (P, 1)
% 
% 	Returns:
% 	----------------------------------------------------
% 	w: learned weighter vector of shape (2D+1, 1)
% 	
% 	Hints:
% 	----------------------------------------------------
% 	You may find "pinv" useful

%% TODO

assert(length(w) == size(F, 1), 'length of w incorrect');