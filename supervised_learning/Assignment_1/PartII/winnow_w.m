function weights = winnow_w(x_train,y_train,weights)
threshold = size(x_train,1);
% winnow_w.m computes the training weights of the winnow
% algorithm. inputs: x_train - training examples, 
% y_train - training labels. weights - initialised to zeros.
% in main code
    % trains quickly - only one epoch needed
    for i = 1:size(x_train,1) % online algorithm one eg at a time        
        y_pred = sign(x_train(i,:)*weights'-threshold);
        if y_train(i) ~= y_pred; % if misclassification, update
            % multiplicative scheme
            weights = weights .* 2.^((y_train(i)-(y_pred>=0)) ...
                                           .*x_train(i,:));
        end
    end
    return 
end


