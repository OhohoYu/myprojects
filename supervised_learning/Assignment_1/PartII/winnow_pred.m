function y_pred = winnow_pred(x_test,w_learned)
% winnow_pred.m - predicts labels for winnow
% classification using learned weights output of winnow_w.m 
% inputs: x_test -> test examples, w_learned-> weights learned from 
% the training set. output: y_pred-> predicted labels for test
% set. 
    % regression vector w defines classifier. f_w = sign(x^T*w)
    threshold = size(x_test,2);
    y_pred = sign(x_test*w_learned'-threshold);
    if y_pred == 0;
        y_pred == -1;
    end
end    
