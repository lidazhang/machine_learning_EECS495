function prob4_9()
% Modified by Stephanie Chang
% Problem 4.9
%
% softmax_grad_demo_hw - when complete reproduces Figure 4.3 from Chapter 4
% of the text

%%% load data 
[X,y] = load_data();

%%% run gradient descent 
[w,error_log] = softmax_newton(X,y);
[wn,archive] = sqr_margin_newton(X,y);

%%% plot everything, pts and lines %%%
plot_all(X',y,w,error_log,archive);


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
    wn = 0.0875*ones(9,1);              % random initial point
    
    % Initializations 
    cycle = 1;
    max_cycle = 30000;
    grad_n = 1;
    archive = [];
    
    N = size(X,1);  %8+1 = 9
    P = size(X,2);  %699

    while  norm(grad_n) > 10^-12 && cycle < max_cycle        
        expr = (X'*wn).*(y); % 699x1 matrix of +ypXp'w
        check = ones(P,1) - expr;
        check(check<0)=0    % Replaces negative 1-ypXp'w values with 0
        grad_n = X*(-2*check.*y)  %(9x699)(699x1) = 9x1     
   
        hess = 2*X*X';
        wn = wn - pinv(hess)*grad_n;
        
        % update iteration count
        if cycle > 1
            error_n = sum(expr(:)<0); 
            archive = cat(1, archive, error_n);
        end
        cycle = cycle + 1;
    end
end

%%% plots everything %%%
function plot_all(X,y,w,error_log,archive)
    % plot softmax optimization
    figure(1)
    s = [1:1:size(error_log,1)]';
    plot (s, error_log);
    hold on
    % plot square margin optimization
    s2 = [1:1:size(archive,1)]';
    plot(s2, archive);
    
    % clean up plot and add info labels
    set(gcf,'color','w');
    axis square
    box off
    axis([2 10 26 32])
    title('Softmax and Square Margin Comparisson')
    xlabel('Iterations','Fontsize',14)
    ylabel('Misclassifications','Fontsize',14)
    set(get(gca,'YLabel'),'Rotation',90)
    legend('softmax','squared margin')
end

%%% loads data %%%
function [X,y] = load_data()
    data = csvread('breast_cancer_data.csv');
    X = data(:,1:end-1);
    y = data(:,end);
end

end
