% extendedtestperceptron.m - Implementation of perceptron algorithm 
% for testing with error analysis functionality. 
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
% does not match the desired target. error_matrix - this variable returns a  
% confusion table presenting which digits have been misclassified with
% which others.  
%
% ESD is a matrix which keeps track of mistakes caused by individual
% scanned digit records
%
function [errors, error_matrix, ESD] = extendedtestperceptron(test,...
                                                  test_kern, alpha, digitno, ESD)
    errors=0;
    % confusion table setup
    error_mat_left = [0 0:digitno-1]'; % leftmost column indexes true labels (0...9)
    % rightmost column tracks total number of times a true label (row) is
    % misclassified. 
    error_mat_right = zeros(digitno+1,1); 
    % error matrix initialised with 0...9 mid-columns keeping track of the number
    % of times a (row) true value has been misclassified for column value. 
    error_matrix = [error_mat_left [0:digitno-1;zeros(digitno,digitno)] ...
                                                      error_mat_right];
    % for error_matrix ignore leftmost and righmost zero in top row
    [m,n] = size(test);    
    for i=1:m
        Y_true=test(i,1);
        weights = zeros(digitno,1); % classifier initialisation
        max_confidence = -1000000000;
        maxi=0;
        kern_row=test_kern(i,:);
        ESD(i,1) = i; 
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
            % if prediction wrong, add mistake
            errors=errors+1;
            % update count of times row value misclassified for column value
            error_matrix(Y_true+2,maxi+2)=error_matrix(Y_true+2,maxi+2)+1;
            % update count of total times given row value is misclassified
            error_matrix(Y_true+2,digitno+2)=error_matrix(Y_true+2,digitno+2)+1;
            % individual scanned digit updated if mistake.
            ESD(i,2) = ESD(i,2)+1;
        end
    end
end