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
K = size(folds,1);
N = size(folds,2); %P/K = 30/3 = 10
X_tmp = ones(K,N);
y_tmp = ones(N,K);
X_train = [];
y_train = [];

for i = 1:K
    if (i == fold_id)   %Validation set
        val_ind = folds(i,:);
        for j = 1:N    %Going through each val in fold
            X_tmp(i,j) = X(1,val_ind(j)); %Build horiz
            y_tmp(j,i) = y(val_ind(j),1); %Build vert
        end
    else                %Training set
        tr_ind = folds(i,:);
        for k = 1:N
            X_tmp(i,k) = X(1,tr_ind(k)); 
            y_tmp(k,i) = y(tr_ind(k),1);
        end
    end
end

X_val = X_tmp(fold_id,:);
y_val = y_tmp(:,fold_id);

for l = 1:K 
    if (l ~= fold_id)
        X_train = cat(2,X_train, X_tmp(l,:)); 
        y_train = cat(1,y_train, y_tmp(:,l)); 
    end
end

assert(length([y_val; y_train]) == length(y), 'Split incorrect');