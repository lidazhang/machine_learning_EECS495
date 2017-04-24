function test()
% Modified by Stephanie Chang
% Problem 4.3c
%
% softmax_grad_demo_hw - when complete reproduces Figure 4.3 from Chapter 4
% of the text

%%% load data 
[X,y] = load_data();

%%% run gradient descent 
[w,error_log] = softmax_newton(X,y);
[wn,archive] = sqr_margin_newton(X,y);

%%% plot everything, pts and lines %%%
plot_all(X',y,w,error_log);


%%%%%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%
%%% gradient descent function for softmax cost/logistic regression %%%
function [w,error_log] = softmax_newton(X,y)
    %%% initialize w0 and make step length %%%
    X = [ones(size(X,1),1) X]';  % use compact notation
    w = 0.008*ones(9,1);              % random initial point
    
    % Initializations 
    iter = 1;
    max_its = 30000;
    grad = 1;
    error_log = [];
    
    N = size(X,1);  %8+1 = 9
    P = size(X,2);  %699
        
    while  norm(grad) > 10^-12 && iter < max_its        
        sig_pow = (X'*w).*(y); % 699x1 matrix of +ypXp'w
        sigma = 1./(ones(P,1)+exp(sig_pow));
        r = -(sigma).*y; %699x1
        grad = X*r;          
   
        hessian = X*((sigma - sigma.^2).*X');
        w = w - pinv(hessian)*grad;
        
        % update iteration count
        if iter > 1
            error = sum(sig_pow(:)<0); 
            error_log = cat(1, error_log, error);
        end
        iter = iter + 1;
    end
end

function [wn,archive] = sqr_margin_newton(X,y)
    %%% initialize w0 and make step length %%%
    X = [ones(size(X,1),1) X]';  % use compact notation
    wn = 0.01*ones(9,1);              % random initial point
    
    % Initializations 
    iter = 1;
    max_its = 30000;
    grad = 1;
    archive = [];
    
    N = size(X,1);  %8+1 = 9
    P = size(X,2);  %699
        
    while  norm(grad) > 10^-12 && iter < max_its        
        sig_pow = (X'*wn).*(y); % 699x1 matrix of +ypXp'w
        error_n = sum(sig_pow(:)<0); 
        sigma = 1./(ones(P,1)+exp(sig_pow));
        r = -(sigma).*y; %699x1
        grad = X*r;          
   
        hessian = X*((sigma - sigma.^2).*X');
        wn = wn - pinv(hessian)*grad;
        
        % update iteration count
        archive = cat(1, archive, error_n);
        iter = iter + 1;
    end
end

%%% plots everything %%%
function plot_all(X,y,w,error_log)
    % plot softmax optimization
    figure(1)
    error_log
    s = [1:1:size(error_log,1)]';
    plot (s, error_log);
    hold on
    % plot square margin optimization
    
    
    % clean up plot and add info labels
    set(gcf,'color','w');
    axis square
    box off
    axis([2 10 26 32])
    title('Softmax and Square Margin Comparisson')
    xlabel('Iterations','Fontsize',14)
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
