function recommender_demo()
% EECS 495: Homework 8
% Problem 9.4
%
% Modified by Stephanie Chang
%-------------------------------------------------------------------------
% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load in data
X = csvread('recommender_demo_data_true_matrix.csv');
X_corrupt = csvread('recommender_demo_data_dissolved_matrix.csv');

K = rank(X);

% run ALS for matrix completion
[C, W] = matrix_complete(X_corrupt, K); 

% plot results
plot_results(X, X_corrupt, C, W)


function [C, W] = matrix_complete(X, K)
% X = 100x200 = NxP
% K = 5
% C = 100x5 = NxK
% W = 5x200 = KxP  
% ---->  YOUR CODE GOES HERE   
    N = size(X,1);
    P = size(X,2);
    C = ones(N,K);
    W = ones(K,P);
    iter = 0
    
    %while(iter>200)
    left1 = sum(diag(C*C'),1); %1x1
    left2 = sum(diag(W*W'),1);
    
    %Find W
    r1_raw = zeros(K,1); %5x1
    for p = 1:P %200    
        for i = 1:N %100
            r1_raw = r1_raw + X(i,p)*C(i,:)';
        end
        r1_mx(:,i) = r1_raw; 
    end
    right1 = sum(r1_mx, 2) %5x1
    W = repmat(right1/left1,[1,P]); %Optimal wp
    
    %Find C
    r2_raw = zeros(200,1);
    for j = 1:P %200
        for n = 1:N %100
            r2_raw = r2_raw + X(n,j)*W(j,:)'
        end    
        r2_mx(:,j) = r2_raw;
    end
    right2 = sum(r2_mx,2)
    cn = right2/left2
    %end
end

function plot_results(X, X_corrupt, C, W)

    gaps_x = [1:size(X,2)];
    gaps_y = [1:size(X,1)];
    
    % plot original matrix
    subplot(1,3,1)
    imshow(X,[])
    colormap hot
    colorbar
    set(gca,'XTick',gaps_x)
    set(gca,'YTick',gaps_y)
    set(gca,'CLim',[0, max(max(X))])
    title('original')
    set(gcf,'color','w');

    % plot corrupted matrix
    subplot(1,3,2)
    imshow(X_corrupt,[])
    colormap hot
    colorbar
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    set(gca,'CLim',[0, max(max(X))])
    title('corrupted')
    set(gcf,'color','w');

    % plot reconstructed matrix
    hold on
    subplot(1,3,3)
    imshow(C*W,[])
    colormap('hot');
    colorbar
    set(gca,'XTick',gaps_x)
    set(gca,'YTick',gaps_y)
    set(gca,'CLim',[0, max(max(X))])
    RMSE_mat = sqrt(norm(C*W - X,'fro')/prod(size(X)));
    f = ['RMSE-ALS = ',num2str(RMSE_mat),'  rank = ', num2str(rank(C*W))];
    title(f)
    set(gcf,'color','w');

end


end

