function y_pred = least_squares_pred(x_test,w_learned)
% least_squares_pred.m - predicts labels for LS regression
% classification using learned weights output of least_squares_w.m 
% inputs: x_test -> test examples, w_learned-> weights learned from 
% the training set. output: y_predict -> predicted labels for test
% set. 
    % regression vector w defines classifier. f_w = sign(x^T*w)
    y_pred = sign(x_test*w_learned); 
    if y_pred == 0;
        y_pred = -1;  
    end
end