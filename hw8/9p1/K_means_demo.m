function K_means_demo()
% EECS 495: Homework 8
% Problem 9.1
%
% Modified by Stephanie Chang

% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load data
X = csvread('kmeans_demo_data.csv');

C0 = [0,0;-.5,.5];     % PART A: unsuccessful initial centroid locations
%C0 = [0,0; 0,.5];     % PART B: successful initial centroid locations

% run K-means
K = size(C0,2);
[C, W] = your_K_means(X, K, C0);

% plot results
plot_results(X, C, W, C0);


function [C, W] = your_K_means(X, K, C0)

% ----->  YOUR CODE GOES HERE  
% X = KxP = 2x21 
% K = 2, P = 21
% C0 = 2x2
    P = size(X,2); 
    C = C0;
    W = zeros(K,P);
    iter = 1;
    
    while (iter < 20)
        % Centroids
        c1 = repmat(C(:,1), [1,P]); %1st col of C0 -> 2x21 
        c2 = repmat(C(:,2), [1,P]); %2nd col of C0 -> 2x21

        d1 = sum((c1 - X).^2,1); %min distance from centroid 1, 1x21
        d2 = sum((c2 - X).^2,1); %min distance from centroid 2, 1x21

        d_c1 = d1 - d2;
        d_c2 = d2 - d1;

        %Replacing negative values with 1, positive values with 0
        d_c1(d_c1>0) = 0; % w1
        d_c1(d_c1<0) = 1;    
        d_c2(d_c2>0) = 0; % w2
        d_c2(d_c2<0) = 1;

        W = [d_c1; d_c2];
        
        for i = 1:P
            if d_c1(i) == 1
                cluster1(:,i) = X(:,i);
            else
                cluster2(:,i) = X(:,i);
            end
        end
        
        C = [(1/nnz(d_c1))*sum(cluster1,2), (1/nnz(d_c2))*sum(cluster2,2)];
        iter = iter + 1;
        
    end
end


function plot_results(X, C, W, C0)
    
    K = size(C,2);
    
    % plot original data 
    subplot(1,2,1)
    scatter(X(1,:),X(2,:),'fill','k');
    title('original data')
    axis([0 1 0 1]);
    set(gcf,'color','w');
    set(gca,'XTick',[]);
    set(gca,'YTick',[]);
    axis square
    axis([-.5 .5 -.5 .5]);
    box on
    hold on
    colors = [0 0 1;
              1 0 0;
              0 1 0;
              1 0 1
              1 1 0
              0 1 1];
          
    for k = 1:K  
        scatter(C0(1,k),C0(2,k),100,'x','MarkerFaceColor',colors(k,:),'MarkerEdgeColor',colors(k,:))
        hold on
    end      
          
                 
    % plot clustered data 
    subplot(1,2,2)
    for k = 1:K
        ind = find(W(k,:) == 1);
        scatter(X(1,ind),X(2,ind),'fill','MarkerFaceColor',colors(k,:),'MarkerEdgeColor',colors(k,:));
        hold on
    end
    
    for k = 1:K  
        scatter(C(1,k),C(2,k),100,'x','MarkerFaceColor',colors(k,:),'MarkerEdgeColor',colors(k,:))
        hold on
    end  
    title('clustered data')
    axis([-.5 .5 -.5 .5]);
    set(gcf,'color','w');
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);
    axis square
    box on
end

end



