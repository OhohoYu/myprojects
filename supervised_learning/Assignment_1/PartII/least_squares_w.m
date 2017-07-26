 function weights=least_squares_w(x_train,y_train)
% least_squares_w.m computes the weights for LS regression
% classification. inputs: x_train - training examples, 
% y_train - training labels
    % may use pinv to compute w of consistent minimal norm
    % efficiency of implementation not focus
    weights = pinv(x_train)*y_train;
 end
