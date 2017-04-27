function softmax_scores = softmax(X, W)
% Return the softmax scores, you can refer to "softmax function" on Wikipedia
% X: N by 785 matrix
% W: 785 by 10 matrix
% softmax_scores: N by 10 matrix
% hint: you may find subtract_max_score()(in this file), exp(), sum(A, 2), and repmat(B, [1, 10]) useful

%% TODO



end

function scores_out = subtract_max_score(scores_in)
% Subtract the max scores of each data point to avoid exponential overflow issue
% scores_in: N by 10 matrix
% score_out: N by 10 matrix
% hint: you may find max(A,[],2) and repmat(B, [1, 10]) useful

%% TODO
max_scores = max(scores_in, [], 2); %Nx1 vector of max values in each row
max_scores_mx = repmat(max_scores, [1,10]); %Nx10 mx with repeated val of max_scores
scores_out = scores_in - max_scores_mx;
end