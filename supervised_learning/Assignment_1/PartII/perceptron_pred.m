function y_pred = perceptron_pred(x_test,w_learned)
% perceptron_pred.m - predicts labels for perceptron
% classification using learned weights output of least_perceptron_w.m 
% inputs: x_test -> test examples, w_learned-> weights learned from 
% the training set. output: y_predict -> predicted labels for test
% set. 
y_pred = sign(x_test*w_learned');
if y_pred == 0;
    y_pred = -1;
end