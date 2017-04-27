function y_pred = predict(W, X)
% Calculate the scores in each class and predict the class label
% W: weight matrix, 785 by 10
% X: N by 785
% hint: you may find max(A, [], 2) very useful

%% TODO
A = X*W; %bj+Xp'W
[argmax, y_pred] = max(A, [], 2); %Extracting biggest value from each row

end
