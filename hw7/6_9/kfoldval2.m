% EECS 495: Homework 7
% Problem 6.9
%
% Modified by Stephanie Chang
%
%
% This file is associated with the book
% "Machine Learning Refined", Cambridge University Press, 2016.
% by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

% parameters to play with
poly_degs = 1:8;    % range of poly models to compare

% load data 
[X,y] = load_data();

num_fold = 3; num_degree = 8;
[fold_tr, fold_val] = random_split(length(y), num_fold); %Uneven splitting
sum_etrain = zeros(num_degree,1);
sum_eval = zeros(num_degree,1);

% perform feature transformation + classification
 [X_train1, X_train2, X_train3, y_train1, y_train2, y_train3...
    X_val1, X_val2, X_val3, y_val1, y_val2, y_val3] = split_data(X, y, fold_tr, fold_val);

% Fold1
[etrain_tmp, eval_tmp] = classify(X_train1,y_train1,X_val1,y_val1,poly_degs);
sum_etrain = sum_etrain + etrain_tmp;
sum_eval = sum_eval + eval_tmp;

% Fold2
[etrain_tmp, eval_tmp] = classify(X_train2,y_train2,X_val2,y_val2,poly_degs);
sum_etrain = sum_etrain + etrain_tmp;
sum_eval = sum_eval + eval_tmp;

% Fold3
[etrain_tmp, eval_tmp] = classify(X_train3,y_train3,X_val3,y_val3,poly_degs);
sum_etrain = sum_etrain + etrain_tmp;
sum_eval = sum_eval + eval_tmp;

avg_etrain = (1/num_degree).*sum_etrain;
avg_eval = (1/num_degree).*sum_eval;

% plot training errors for visualization
plot_errors(poly_degs, avg_etrain, avg_eval)
    
%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%%%%

function [fold_tr, fold_val] = random_split(P, K) 
% Parsing points into 3 uneven groups

values = randperm(P,P);
fold_tr = reshape(values(1:2*(P-1)/K),[K-1,(P-1)/K]); %Training folds have 33 #s
fold_val = reshape(values((2*(P-1)/K)+1:P),[1,(P-1)/K + 1]); %Validation fold has 34 #s
end

function [X_train1, X_train2, X_train3, y_train1, y_train2, y_train3...
    X_val1, X_val2, X_val3, y_val1, y_val2, y_val3] = split_data(X, y, fold_tr, fold_val)
% Getting actual training and validation values using fold indices
K = size(fold_tr, 1); %2
N = size(fold_tr, 2); %33
K_val = size(fold_val, 1); %1
N_long = size(fold_val,2); %34

    X_tmp = ones(K, N); %2x33
    y_tmp = ones(N, K); %33x2
    X_tmp_val = ones(K_val, N_long); %1x34
    y_tmp_val = ones(N_long ,K_val); %34x1

    X = X'; %2x100

    % Making Xtrain and ytrain, fold 1
    for i = 1:K
        tr_ind = fold_tr(i,:); %1x33
        for k = 1:N
            X_tmp(i,k) = X(1,tr_ind(k)); %2x33
            y_tmp(k,i) = y(tr_ind(k),1); %33x2
        end
    end

    X_train1 = cat(2, X_tmp(1,:), X_tmp(2,:)); %1x66
    y_train1 = cat(1, y_tmp(:,1), y_tmp(:,2)); %66x1

    % Making Xval and yval
    for j = 1:K_val
    test_ind = fold_val(j,:);
        for l = 1:N_long
            X_tmp_val(j,l) = X(1,test_ind(l)); %1x34
            y_tmp_val(l,j) = y(test_ind(l),1); %34x1
        end
    end

    X_val1 = X_tmp_val;
    y_val1 = y_tmp_val;
    
    % fold2
    X_train2 = cat(2,X_tmp(2,:), X_tmp_val); %1x67
    y_train2 = cat(1,y_tmp(:,2), y_tmp_val); %67x1 
    X_val2 = X_tmp(1,:); %1x33
    y_val2 = y_tmp(:,1); %33x1
    
    % fold3
    X_train3 = cat(2, X_tmp(1,:), X_tmp_val); %1x67
    y_train3 = cat(1, y_tmp(:,1), y_tmp_val); %67x1
    X_val3 = X_tmp(2,:);
    y_val3 = y_tmp(:,2);

end %end function

function [errors_tr, errors_val] = classify(X_train,y_train,X_val,y_val,poly_degs)  

    errors_tr = [];
    errors_val = [];
    error_min = 1000;
    
    % solve for weights and collect errors
    for i = 1:length(poly_degs)
        % generate features
        poly_deg = poly_degs(i);         
        F_train = poly_features(X_train,poly_deg);
        F_val = poly_features(X_val,poly_deg);
        
        % run logistic regression
        w = softmax_grad(F_train',y_train);
        
        % calculate training errors
        resid_tr = evaluate(F_train',y_train,w);    
        errors_tr = [errors_tr; resid_tr];
        
        % calculate testing errors
        resid_val = evaluate(F_val',y_val,w);
        errors_val = [errors_val; resid_val];
    end
    
end

    
%%% builds (poly) features based on input data %%%
function F = poly_features(X,D)

x_expand = ones(D+1,1)*X; % (D+1)xP, x1s all in first col
F = x_expand.^(linspace(0,D,D+1)'); 

end

%%% plots points for each fold %%%
function plot_pts(X,y)
    
    % plot training set
    ind = find(y == 1);
    red =  [ 1 0 .4];

    plot(X(ind,1),X(ind,2),'o','Linewidth',2.5,'MarkerEdgeColor',red,'MarkerFaceColor','none','MarkerSize',7)
    hold on
    ind = find(y == -1);
    blue =  [ 0 .4 1];
    plot(X(ind,1),X(ind,2),'o','Linewidth',2.5,'MarkerEdgeColor',blue,'MarkerFaceColor','none','MarkerSize',7)
   
end

%%% plots training errors %%%
function plot_errors(poly_degs, error_tr, error_val)
    figure(2)
    hold on
    h2 = plot(1:max(poly_degs), error_tr,'yv--');
    hold on
    h3 = plot(1:max(poly_degs), error_val,'bv--');
    hold on
    plot(1:max(poly_degs),error_tr,'yv--')
    plot(1:max(poly_degs),error_val,'bv--')
    title('Average Error vs. Polynomial Basis Degree')
    legend('Training error','Validation error','Location','northeast')

    % clean up plot
    set(gcf,'color','w');
    box on
    xlabel('D','Fontsize',18,'FontName','cmr10')
    ylabel('error','Fontsize',18,'FontName','cmr10')
    set(get(gca,'YLabel'),'Rotation',90)
    set(gcf,'color','w');
    box off
    axis square
end

%%% loads and plots labeled data %%%
function [X,y] = load_data()
      
    % load data
    data =  csvread('2eggs_data.csv');
    X = data(:,1:end - 1);
    y = data(:,end);
end

%%% gradient descent for softmax %%%
function w = softmax_grad(Fbar,y) 
    
    % initialize
    %Fbar with 1's in col1 
    Fbar = Fbar';
    w = randn(size(Fbar,1),1); %66x1
    alpha = 10^-1;

    % precomputations
    grad = 1;
    n = 1; % iterations
    nmax = 10^6;
    N = size(Fbar,1); %2
    P = size(Fbar,2); %66

    %%% main %%%
    while n <= nmax && norm(grad) > 10^-5

        % prep gradient for logistic objective
        r = sigmoid(-(Fbar'*w).*y).*y;
        grad = -Fbar*r;

        % take step
        w = w - alpha*grad;
        n = n + 1;
    end
end

%%% sigmoid function for use with grad descent %%%
function t = sigmoid(z)
    t = 1./(1+exp(-z));
end

%%% evaluates error of a learned model %%%
function score = evaluate(A,b,w) %X', y, w
    s = A*w;
    ind = find(s > 0); % Get positions of positive #
    s(ind) = 1;
    ind = find(s <= 0); % Get positions of neg #
    s(ind) = -1;
    t = s.*b;
    ind = find(t < 0);
    t(ind) = 0;
    score = 1 - sum(t)/numel(t);

end