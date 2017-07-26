function generalisation_err = accuracy(x_train,y_pred)
% accuracy.m computes the generalisation error between 
% y_train and y_pred for a given classifier. 
% inputs -> x_train: the training examples, recall that their
% true labels are given by x_train(:,1) - first col. 
% y_pred -> predicted labels for examples
% output -> generalisation_err for a given classifier.
    generalisation_err = mean(x_train(:,1)~=y_pred);
end

