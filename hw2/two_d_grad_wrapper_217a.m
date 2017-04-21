%Problem 2.17 Part A
%Modified by Stephanie Chang

function two_d_grad_wrapper_217a()
% two_d_grad_wrapper.m is a toy wrapper to illustrate the path
% taken by gradient descent depending on the learning rate (alpha) chosen.
% Here alpha is kept fixed and chosen by the use. The corresponding
% gradient steps, evaluated at the objective, are then plotted.  The plotted points on
% the objective turn from green to red as the algorithm converges (or
% reaches a maximum iteration count, preset to 50).
% (nonconvex) function here is 
%
% g(w) = log(1+exp(w'*w))

%%% runs everything %%%
run_all()

%%%%%%%%%%%% subfunctions %%%%%%%%%%%%
%%% Part A. Performs Gradient Descent to Find Stationary Point %%%%
    function [w,in,out] = grad_descent(alpha,w)

    % initializations
    grad_stop = 10^-5;
    %max_its = 50;
    iter = 1;
    grad = 1;
    in = [w]; %2x1
    out = [log(1+exp(w'*w))]; %The function
    % main loop
    while norm(grad) > grad_stop %&& iter <= max_its
        % take step
        grad = (2*exp(w'*w)/(1+exp(w'*w)))*w; %2x1   
        w = w - alpha*grad; %wnew = wold - g'(wold)/g''(wold)

        % update containers
        in = [in, w];
        out = [out, log(1+exp(w'*w))];

        % update stopers
        iter = iter + 1;
    end
end

function run_all()
    % dials for the toy
    alpha = 0.01;     % step length/learning rate (for gradient descent). Preset to alpha = 10^-3

    for j = 1:2
        x0 = [1;1];         % initial point (for gradient descent)
        if j == 2
            x0 = [.85;.85];
            alpha = 3*10^-3;
        end
        %%% perform gradient descent %%%
        [x,in,out] = grad_descent(alpha,x0);

        %%% plot function with grad descent objective evaluations %%%
        hold on
        plot_it_all(in,out)
    end
    
    x
end

%%% plots everything %%%
function plot_it_all(in,out)
    % print function
    [A,b] = make_fun();
    
    % print steps on surface
    plot_steps(in,out,3)
    set(gcf,'color','w');
end

%%% Part B. Surface plot  %%%
function [A,b] = make_fun()
    range = 1.15;                     % range over which to view surfaces
    [a1,a2] = meshgrid(-range:0.04:range);
    a1 = reshape(a1,numel(a1),1);
    a2 = reshape(a2,numel(a2),1);
    A = [a1, a2];
    A = (A.*A)*ones(2,1);
    b = log(1+exp(A)); %log(1+exp(w'*w))
    r = sqrt(size(b,1));
    a1 = reshape(a1,r,r);
    a2 = reshape(a2,r,r);
    b = reshape(b,r,r);
    h = surf(a1,a2,b)
    az = 35;
    el = 60;
    view(az, el);
    shading interp

    
    xlabel('w_1','Fontsize',18,'FontName','cmmi9')
    ylabel('w_2','Fontsize',18,'FontName','cmmi9')
    zlabel('g','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'ZLabel'),'Rotation',0)
    set(gca,'FontSize',12);
    box on
    colormap parula
end

% plot descent steps on function surface
function plot_steps(in,out,dim)
    s = (1/length(out):1/length(out):1)';
    colorspec = [ones(length(out),1) ,zeros(length(out),1),flipud(s)];
    width = (1 + s)*5;
    if dim == 2
        for i = 1:length(out)
            hold on
            plot(in(1,i),in(2,i),'o','Color',colorspec(i,:),'MarkerFaceColor',colorspec(i,:),'MarkerSize',width(i));
        end
    else % dim == 3
        for i = 1:length(out)
            hold on
            plot3(in(1,i),in(2,i),out(i),'o','Color',colorspec(i,:),'MarkerFaceColor',colorspec(i,:),'MarkerSize',width(i));
        end
    end
end

end
