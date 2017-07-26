function  weights = perceptron_w(x_train,y_train,weights)
% perceptron_w.m computes the training weights of the perceptron
% algorithm. inputs: x_train - training examples, 
% y_train - training labels. weights - initialised to ones.
    for j = 1:size(x_train,1); % online algorithm. one eg at time
        % only one epoch needed - trains quickly 
        y_pred = sign(x_train(j,:)*weights'); % y_pred
        if y_train(j)*y_pred <= 0; % if misclassification, update
            weights = weights + y_train(j)*x_train(j,:); % weight update
        end
    end
    return 
end