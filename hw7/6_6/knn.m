% EECS 495: Homework 7
% Problem 6.6
% 
% Written by Stephanie Chang

function knn()
    [Xtrain,ytrain] = load_data();
    k = 10; % # of neighbors to sample    
    Xtest = test_data(); %Nx2
    ytest = [];
    N = size(Xtrain,1);
    
    for i = 1:size(Xtest, 1)
        % Find the Euclidean distance betwen each training pt and 1 test pt
        diff = repmat(Xtest(i,:),[N,1]) - Xtrain;
        euclDist = sqrt(sum(diff.^2, 2));   
    
        % Find k closest neighbors & average tags
        [closest, index] = sort(euclDist); 
        ytest_tmp = ytrain(index(:));
        ytest_avg = sum(ytest_tmp(1:k),1)/k;  
        if (ytest_avg >= 0)
            ytest(i) = 1;
        else 
            ytest(i) = -1;
        end
        
    end % end for loop
            
    plot_data(Xtrain,Xtest,ytest,k);
end

function [Xtrain, ytrain] = load_data()
    data = csvread('knn_data.csv');
    Xtrain = data(:,1:end-1); %30x2
    ytrain = data(:,end);     %30x1

    % Replacing 0 tags with -1's
    ytrain(ytrain == 0)=-1;
end

function Xtest = test_data()
    % Creating dense grid of points [-10, 10]
    x1test = 0:0.01:10;
    x2test = 0:0.01:10;
    [x1test, x2test] = meshgrid(x1test, x2test);
    Xtest = [x1test(:), x2test(:)]; % Nx2 List of points in grid    
end

function plot_data(Xtrain,Xtest,ytest,k)
    figure(1)
    hold on
    % Colored regions
    red = find(ytest == 1)';
    blue = find(ytest == -1)';
    scatter(Xtest(red(:),1), Xtest(red(:),2), 'r')
    scatter(Xtest(blue(:),1), Xtest(blue(:),2), 'b');
    
    % data points
    scatter(Xtrain(:,1), Xtrain(:,2),'k','filled');
    str = sprintf('k-NN Classification, k = %d', k);
    title(str)
    xlabel('x1')
    ylabel('x2')
    %axis([1,10,1,10])
end