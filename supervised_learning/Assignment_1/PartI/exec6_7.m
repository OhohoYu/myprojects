% Section 1 Exercise 6
clear all;
clc;
close all;

% The data set is taken out of Exercise 2, Part a: 
% The data set has 600 samples and is 10 dimentional.
ntot = 600;
dim = 10;

% The first 100 samples in the data set is dedicated to the train set and
% the rest is allocated to the test set. 
ntrain = 10;
ntest = ntot - ntrain;

% K_fold is the number of k-fold disjoint sets
K_fold = 5;  

% Initialising an array to the parameter gamma and the number of its elements.    
gamma = 10.^(-6:3);
gamma_size = numel(gamma);

%  w_learned is generated using the normal distribution.
w_learned = randn(dim,1); 

% 600 x-variables are also randomly generated from the normal distrubtion,
x = randn(ntot,dim); 
% An array of noise is generated from the normal distribution.
eps = randn(ntot,1); 

% Random y_variables are generated using y = x*w' + eps.
for i = 1:ntot
   y(i,:) = x(i,:)*w_learned + eps(i,:); 
end

% These traing sets include the first 80 samples of the data set.
x_train = x(1:8,:);
y_train = y(1:8);

% The validation sets include the range 81 to 100th sample of the data set.
x_val = x(9:ntrain,:);
y_val = y(9:ntrain);

% The original training sets include the first 100 samples from the
% data set. The training and the validation sets both make up the original
% trining set.
x_train_original = x(1:ntrain,:);
y_train_original=y(1:ntrain);

% The test data set includes the 101 to 600th sample.
x_test = x(ntrain+1:end,:);
y_test = y(ntrain+1:end);

% The training set is sub-divided into smaller validation sets.
fold_sub = ntrain/K_fold;

% In the following for loop, a validation set is picked and the rest of the
% original training set is then assigned to the training set for the x and
% y variables.
% At each iteration, a different validation set is picked and the process
% is repeated.

for k = 1:K_fold
   x_train_fold = x_train_original(1:fold_sub,:);
   x_val_fold = x_train_original(fold_sub+1:end,:);
   y_train_fold = y_train_original(1:fold_sub,:);
   y_val_fold = y_train_original(fold_sub+1:end,:);
   x_train_original = circshift(x_train_original,[fold_sub 0]); 
   y_train_original = circshift(y_train_original,[fold_sub 0]);
end


% The aim of this for loop is to predict the value of y and calculate its
% Mean Squared Error from the actual value of y. This process is repeated
% 200 times.
mse_test_original = zeros(1,gamma_size);
mse_train_original = zeros(1,gamma_size);
mse_val = zeros(1,gamma_size);
iteration = 200;
train_error = zeros(1,iteration); 


for j = 1:iteration
    
%     Ridge regression is implementend:
    for k = 1:K_fold
   x_train_fold = x_train_original(1:fold_sub,:);
   x_val_fold = x_train_original(fold_sub+1:end,:);
   y_train_fold = y_train_original(1:fold_sub,:);
   y_val_fold = y_train_original(fold_sub+1:end,:);
   x_train_original = circshift(x_train_original,[fold_sub 0]); 
   y_train_original = circshift(y_train_original,[fold_sub 0]);


    for l = 1:gamma_size
        

        %  Ridge regression is done on the training folds and the respective MSE is
        %  calculated.       
            w_fold = w_ridge(x_train_fold,y_train_fold,gamma(l));
            mse_train_fold(k,l) = MSE(x_train_fold,y_train_fold,w_fold);
            mse_val_fold(k,l) = MSE(x_val_fold,y_val_fold,w_fold);
            mse_test_fold(k,l) = MSE(x_test, y_test, w_fold);
        
    end
            % The avarage MSE for the training and validation folds is then calculated       
        mse_train_original = sum(mse_train_fold)./K_fold;
        mse_val_avg = sum(mse_val_fold)./K_fold;
        mse_test_avg = sum(mse_test_fold)./K_fold;
        
    end
   
    
    % The optimal gamma that gives minimum five fold validation error
    [N3,D3] = min(mse_val_avg);
    kfold_val_error(j) = gamma(D3);
    
    % MSE using optimum gamma for five fold validation set:
    w_learned = w_ridge(x_train_original,y_train_original,kfold_val_error(j));
    mse_opt(j) = MSE(x_test,y_test,w_learned);
   
end



% A graph of log of gamma against the MSEs' is then plotted.


 hold on
 plot(log(gamma),mse_train_original)
 plot(log(gamma),mse_val_avg)
 plot(log(gamma),mse_test_avg,'green')
 hold off;
 
  title('Main Training set of 10 Samples')
legend('MSE Main Training Set','MSE Test Set','MSE Validation Set')
 xlabel('log Gamma');
 ylabel('MSE');

% Question 7) part c)
   if ntrain == 100
       mse_optimum_train_avg_hundred = mean(mse_train_original)
       mse_optimum_val_avg_hundred = mean(mse_val_avg)
       mse_optimum_kfold_avg_hundred = mean( mse_test_avg)
   elseif ntrain == 10
       mse_optimum_train_avg_ten = mean(mse_train_original)
       mse_optimum_val_avg_ten = mean(mse_val_avg)
       mse_optimum_kfold_avg_ten = mean( mse_test_avg)
   end