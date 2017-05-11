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

r_mx = sum(F*y,1); %(2D+1,1)
l_mx = F*F'; %(2D+1, 2D+1)
w = pinv(l_mx)*r_mx; %(2D+1,1)

assert(length(w) == size(F, 1), 'length of w incorrect');