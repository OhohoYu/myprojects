clear all
%
% COMPGI07 ASSIGNMENT 2 - QUESTION 3
%
% train = load('dtrain123.dat');
% test = load('dtest123.dat');
%
train = load('ziptrain.dat');
test = load('ziptest.dat');
%
% number of classes/nepochs
%
nepochs = 4; 
digitno = 10;
max_degree = 7;
%
% divide train further into train and validation
%
[m,n] = size(train);
split = round((2/3)*m); % 2:1 train:val split
train_set = train(1:split,:);
val_set = train(split+1:end,:);
[mtrain, ntrain] = size(train_set);
[mval,nval] = size(val_set);
%
fprintf('Using hold-out to compute optimum polynomial degree: \n')
%
% test error table for different degrees
holdout_degree=[2:max_degree; ones(1,max_degree-1)]; 
ESD1 = zeros(mval,2);
%     
%loop for different value of degree
for degree=2:max_degree
    fprintf('Computing validation error for a degree of %d.\n', degree)
    % good for performance - remove kernel computation from epoch iteration
    train_kern=poly_kernel(train_set(:,2:end),train_set(:,2:end), degree);
    val_kern=poly_kernel(val_set(:,2:end),train_set(:,2:end), degree);
    alpha = zeros(digitno, mtrain);
    % iterate over epochs
    for epoch=1:nepochs        
        [training_errors, alpha]=trainperceptron(train_set,train_kern,alpha,digitno);
        fprintf('Training - Epoch %i: %i mistakes out of %i examples:\n', ...
                  [epoch training_errors mtrain])          
        [test_errors, error_matrix, ESD1]=extendedtestperceptron(val_set, ...
                                                   val_kern,alpha, digitno, ESD1);
        percentage_error = test_errors*100/mval;
        fprintf('Testing - Epoch %i: Test error is %f percent.\n\n', [epoch percentage_error])
        % once the final epoch has been reached, print out confusion table
        if epoch==nepochs
            holdout_degree(2,degree-1)=percentage_error;
            disp(error_matrix)   
        end        
    end
end

% print out table of test error vs degree
disp(holdout_degree)

% MODEL SELECTION

% graph of polynomial kernel degree plotted against validation error. 
figure
plot(holdout_degree(1,:),holdout_degree(2,:))
% optimal polynomial kernel degree is d=4.
[opt_err optimal_degree]= min(holdout_degree(2,:));
optimal_degree = optimal_degree+1;
fprintf('Model selection completed. Optimal degree is %d\n', optimal_degree)
%
% We now retrain our classifier on the dataset train_set + val_set 
% with the chosen parameter d=4. Hold out method.  
%
holdout_degree_final=[2:max_degree; ones(1,max_degree-1)]; 
%
fprintf('Retrain classifier on train_set + val_set, d=4.\n')
original_train = train;
[mot,not] = size(original_train);
[mt,nt] = size(test);
min_test_err = 100000000; 
%
% for d=4
ESD2 = zeros(mt,2);
fprintf('Computing test error with a polynomial kernel degree of 4.\n')
% Kernel computations
original_train_kern=poly_kernel(original_train(:,2:end),...
                                 original_train(:,2:end), optimal_degree);
test_kern=poly_kernel(test(:,2:end),original_train(:,2:end), optimal_degree);
alpha_final = zeros(digitno, mot);
% iterate over epochs
for epoch=1:nepochs        
    [tr_errors_final, alpha_final]=trainperceptron(original_train,...
                                      original_train_kern, alpha_final, digitno);
    fprintf('Training - Epoch %i: %i mistakes out of %i examples:\n', ...
                              [epoch tr_errors_final mot])          
    [test_errors_final, final_error_matrix, ESD2]=extendedtestperceptron(test, ...
                                          test_kern,alpha_final, digitno, ESD2);
    percentage_error_final = test_errors_final*100/mt;
    fprintf('Testing - Epoch %i: Test error is %f percent.\n\n', [epoch ...
                                                 percentage_error_final]) 
    if percentage_error_final < min_test_err
        min_test_err = percentage_error_final;
        opt_epoch = epoch;
    end   
end
fprintf('The percentage test error after %d epochs is %f percent.\n', ...
                                   epoch, percentage_error_final)
fprintf('A min test error of %f percent is obtained after %d test epochs.\n\n', ...
                                   min_test_err, opt_epoch)

disp('Confusion table for test set, d=4, 4 epochs: ')
disp(final_error_matrix)

y_true_classes = test(:,1);
occurrence_count = zeros(1,digitno);
for i=1:digitno
    occurrence_count(i) = sum(y_true_classes==y_true_classes(i));
end    

relative_error = zeros(4,digitno);
relative_error(1,:) = 0:digitno-1;
relative_error(2,:) = (final_error_matrix(2:digitno+1,digitno+2))';
relative_error(3,:) = occurrence_count;
relative_error(4,:) = relative_error(2,:)./relative_error(3,:)*100;

disp('Test percentage errors for specific classes: ')
disp(relative_error)

%
% our aim is to find the most difficult to recognise scanned digits. we now
% loop over all polynomial degrees and epochs again, this time over the
% full training set and full test set. This final step is performed to 
% keep track of the specific most misclassified scanned digits. in order to
% do this, we add variable ESD to the perceptron_test function.     
%
% loop for different value of degree
ESD3 = zeros(mt,2);
for degree=2:max_degree
    fprintf('Computing the most misclassified scanned digits for degree of %d.\n',... 
                                                                    degree)
    % good for performance - remove kernel computation from epoch iteration  
    original_train_kern=poly_kernel(original_train(:,2:end),...
                                 original_train(:,2:end), degree);
    test_kern=poly_kernel(test(:,2:end),original_train(:,2:end), degree);
    alpha_final = zeros(digitno, mot);
    % iterate over epochs
    for epoch=1:nepochs
        [tr_errors_final, alpha_final]=trainperceptron(original_train,...
                                      original_train_kern, alpha_final, digitno);
        [test_errors_final, final_error_matrix, ESD3]=extendedtestperceptron(test, ...
                                          test_kern,alpha_final, digitno, ESD3);         
    end
end
% ESD3 returned - table with all specific scanned digit indexes and their
% respective number of mistakes.

% top_number specifies the number of top entries that are displayed
top_number=5;
% sort ESD in descending order
[mistake_rank, hardest_to_recognize] = sort(ESD3(:,2),1,'descend');
fprintf('The 5 hardest-to-recognize digits over 6 degree iterations (4 epochs each) are: \n')
% table with scanned digit entries and their mistakes in descending order
hard2rec = [hardest_to_recognize'; mistake_rank'];
% table only featuring top 5 most misclassified scanned digit records. 
top5 = hard2rec(:,1:top_number);
disp(top5);

% display printoffs
% the true label of the printoffs is displayed in the command window

% these are plotted after inspecting the top5 to find entries
figure; 
subplot(2,3,1)
fprintf('True label of digit is %i\n\n', test(18,1))
imagesc(reshape(test(18,2:end), 16, 16)'); colormap 'gray';
subplot(2,3,2)
fprintf('True label of digit is %i\n\n', test(123,1))
imagesc(reshape(test(123,2:end), 16, 16)'); colormap 'gray';
subplot(2,3,3)
fprintf('True label of digit is %i\n\n', test(135,1))
imagesc(reshape(test(135,2:end), 16, 16)'); colormap 'gray';
subplot(2,3,4)
fprintf('True label of digit is %i\n\n', test(199,1))
imagesc(reshape(test(199,2:end), 16, 16)'); colormap 'gray';
subplot(2,3,6)
fprintf('True label of digit is %i\n\n', test(234,1))
imagesc(reshape(test(234,2:end), 16, 16)'); colormap 'gray';

