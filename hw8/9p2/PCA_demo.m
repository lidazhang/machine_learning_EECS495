function PCA_demo()
% EECS 495: Homework 8
% Problem 9.2
% 
% Modified by Stephanie Chang
%-------------------------------------------------------------------------
% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% load in data
X = csvread('PCA_demo_data.csv');
n = size(X,1);
means = repmat(mean(X),n,1);
X = X - means;  % center the data
X = X';

K = 1;  

% run PCA 
[C1, W1] = PCA_svd(X, K);
[C2, W2] = PCA_opt(X, K);

% plot_results
plot_results(X, C1, 1)
plot_results(X, C2, 2)

function [C, W] = PCA_svd(X, K)
% X = 2x150 = NxP
% K = 1
% ---->  YOUR CODE GOES HERE 
    
    [U,S,V] = svd(X); %U = 2x2, S = 2x150, V = 150x150
    C = U(:,K)*S(K,K); %2x1
    W = V(:,K)'; %1x150
   
end


function [C, W] = PCA_opt(X, K)
% X = 2x150 = NxP
% K = 1
% ---->  YOUR CODE GOES HERE 
N = size(X,1);
P = size(X,2);

%Initializations
C = rand([N,K]);  
W = ones([K,P]); %W(k-1)
c_check = 100;
w_check = 100;
iter = 1;

   while(iter<50)
        C = X*W'*pinv(W*W'); %Ck
        W = pinv(C'*C)*C'*X;
        iter = iter + 1;
   end
end

function plot_results(X, C, part)
    figure(part)
    
    % Print points and pcs
    subplot(1,2,1)
    for j = 1:n
        hold on
        scatter(X(1,:),X(2,:),'fill','b')
    end
    hold on
    %s = C(1,1):0.001:-C(1,1);
    s = -0.5:.001:.5;
    m = C(2,1)/C(1,1);
    plot(s,m*s,'k','LineWidth',2.5)
    xlabel('b_1','Fontsize',14,'FontName','cmr10')
    ylabel('b_2','Fontsize',14,'FontName','cmr10')

     if part == 1
        str = sprintf('Part A: SVD');
     elseif part == 2
         str = sprintf('Part B: Alternate Optimization');
     end
    title(str);
    
    axis([-0.5 0.5 -0.5 0.5])    % Set viewing axes
    axis square

    % Plot projected data
    subplot(1,2,2)

    X_proj = C*((C'*C)\(C'*X));
    for j = 1:n
        hold on
        scatter(X_proj(1,:),X_proj(2,:),'fill','b')
    end
    axis([-0.5 0.5 -0.5 0.5])    % Set viewing axes
    xlabel('b_1','Fontsize',14,'FontName','cmr10')
    ylabel('b_2','Fontsize',14,'FontName','cmr10')

    axis square
    set(gcf,'color','w');

end

end


    
    








