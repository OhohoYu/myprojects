% testperceptron.m - Implementation of perceptron algorithm for testing.
% very similar to trainperceptron.m, but only requires the prediction
% step for each example in the test set. Update step is not performed. 
%
% train -> a set of testing inputs X (includes (x_1, y_1)...(x_m, y_m 
% )). test_kern - a polynomial kernel formed by the test inputs
% (x_1,...,x_m).  weight_mat - weight vector inherited from training phase
% into a majority network of perceptrons to separate digitno classes.
% digitno -> number of classes/ different digits available
% 
% Outputs: errors - this variable returns the total number of mistakes made
% during the testing process i.e.the number of times the predicted output 
% does not match the desired target.  
function [errors] = testperceptron (test, test_kern, alpha, digitno)
    errors=0;
    [m,n] = size(test);
    for i=1:m
        Y_true=test(i,1);
        weights = zeros(digitno,1); % classifier initialisation
        max_confidence = -1000000000;
        maxi=0;
        kern_row=test_kern(i,:);
        for j=1:digitno % we test digitno 2-classifiers
            sum=0;
            beta = alpha(j,:); % dummy
            for k=1:length(alpha) % loop over test examples 
                % single kernel function scaled by the term alpha
                sum=sum+ beta(k)*kern_row(k); % product added to sum 
            end
            % test example weights
            weights(j) = sum;  %w_k = sum(alpha_i*K*x_i,...)%
        end
        for l=0:digitno-1
            % update step eliminated
            if weights(l+1)>max_confidence
                maxi=l;
                max_confidence=weights(l+1);
            end
        end
        if maxi~=Y_true;
            errors=errors+1;
        end
    end
end
