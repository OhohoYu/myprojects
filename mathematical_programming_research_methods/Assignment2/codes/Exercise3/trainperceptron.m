% trainperceptron.m - Implementation of the training perceptron algorithm.
%
% Inputs: train - a set of training inputs X (includes (x_1, y_1)...(x_m, y_m 
% )). train_kern - a polynomial kernel formed by the training inputs
% (x_1,...,x_m). alpha - coefficient matrix updates as we cycle through the
% program. digitno - number of different digits available. We generalise
% into a majority network of perceptrons to separate digitno classes. 
% 
% Outputs: errors - this variable returns the total number of errors made
% during the training process i.e.the number of times the predicted output 
% does not match the desired target for selected 2-classifiers.
% alpha - a trained/updated matrix of  coefficients for later use in 
% test steps.

function [errors,alpha]= trainperceptron( train, train_kern, alpha, digitno)
    errors=0;
    [m,n] = size(train);
    for i=1:m % online algorithm operates on single example at a time
        Y_true = train(i,1);  % y_i and rest of train (X_i): input
        weights = zeros(digitno,1); % classifier initialisation
        % single kernel function K(x_t, .) added for each example
        kern_row = train_kern(i,:);  
        max_confidence = -1000000000; % dummy value
        max_w=0; % dummy value
        % classifier training: algorithm generalised for digitno classes. 
        for j=1:digitno % we train digitno 2-classifiers
            sum=0;
            beta = alpha(j,:); % dummy
            for k=1:length(alpha) % loop over training examples 
                % single kernel function scaled by the term alpha
                sum=sum+ beta(k)*kern_row(k); % product added to sum 
            end
            % trained weight for a 2-classifier for a training example
            weights(j) = sum;  %w_k = sum(alpha_i*K*x_i,...)%
        end
        for l=0:digitno-1
            if Y_true==l;
                Y=1;  % examples with desired output class given +ive label
            else
                Y=-1;  % otherwise are given negative label
            end
            if Y*weights(l+1)<=0 % if predicted output does not match  target 
                % incorrect prediction leads to change in the coeff matrix
                if weights(l+1) <=0
                    alpha(l+1,i)=alpha(l+1,i)-(-1); % for Y=-1 add (in this case).
                else
                    alpha(l+1,i)=alpha(l+1,i)-(1);  % for Y=1 subtract
                end
            end
            % check associated confidence of each 2-classifier/weights (k =w)
            if weights(l+1)>max_confidence % if argmax
                max_w=l; % new most confident 2-classifier
                max_confidence=weights(l+1); % new max_confidence
            end
        end    
        if max_w~=Y_true % for selected 2-classifier add to mistake count
            errors=errors+1; % when output does not match target
    end
end
          
        
        
        
        
        
        
        
        
  



