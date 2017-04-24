function prob4_9()
% Modified by Stephanie Chang
% Problem 4.3c
%
% softmax_grad_demo_hw - when complete reproduces Figure 4.3 from Chapter 4
% of the text

%%% load data 
[X,y] = load_data();

%%% run gradient descent 
[w,error_log] = softmax_gradient_descent(X,y);

%%% plot everything, pts and lines %%%
plot_all(X',y,w,error_log);


%%%%%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%
%%% gradient descent function for softmax cost/logistic regression %%%
function [w,error_log] = softmax_gradient_descent(X,y)
    %%% initialize w0 and make step length %%%
    X = [ones(size(X,1),1) X]';  % use compact notation
    w = randn(9,1);              % random initial point
    alpha = 10^-2;               % fixed steplength for all iterations
    
    % Initializations 
    iter = 1;
    error_log = [];
    max_its = 30000;
    grad = 1;
    
    N = size(X,1);  %9
    P = size(X,2);  %699

    while  norm(grad) > 10^-12 && iter < max_its
        % compute gradient
        sig_in = (X'*w).*(y); % 699x1 matrix of ypXp'w
        sigma = 1./(ones(P,1)+exp(sig_in));
        r = -(sigma).*y; %699x1
        grad = X*r;           % YOUR CODE GOES HERE
        w = w - alpha*grad;
        
        error = sum(sig_in(:)<0); %Counting number of misclassifications
        error_log = cat(1, error_log, [iter, error]);
        
        % update iteration count
        iter = iter + 1;
    end
    error_log;
    w
end

%%% plots everything %%%
function plot_all(X,y,w,error_log)
    % plot separator
    plot (error_log(:,1),error_log(:,2));
    
    % clean up plot and add info labels
    set(gcf,'color','w');
    axis square
    box off
    axis([0 1 0 1])
    title('Figure 4.8 Reproduction')
    xlabel('Iteration','Fontsize',14)
    ylabel('Misclassifications','Fontsize',14)
    set(get(gca,'YLabel'),'Rotation',90)
end

%%% loads data %%%
function [X,y] = load_data()
    data = csvread('breast_cancer_data.csv');
    X = data(:,1:end-1);
    y = data(:,end);
end

end
