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
P = size(X,2); %30
bias = ones(1,P); %first row all ones
deg = repmat(linspace(1,D,D)',1,P); %DxP 
fin = repmat(2*pi*X,D,1).*deg; %DxP
fcos = cos(fin);
fsin = sin(fin);
f = ones(size([fcos;fsin])); %(2D,P)
f = (1:2:end,:) = fcos; %Replacing odd rows with cosine
f = (2:2:end,:) = fsin; %Replacing even rows with sine

% for i=1:2*D
%     if mod(i,2) == 1        %Odd 
%         f = fcos(i,:); 
%     else                    %Even
%         f = fsin(i,:);
%     end
% end

F = cat(1,bias,f); %Adding bias to top row

assert(size(F, 1)==2*D+1, 'degree incorrect');
assert(size(F, 2)==size(X, 2), 'number of data points incorrect');