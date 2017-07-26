% Supervised Learning 1) Section 1) Exercise 9) Part a)

load ('boston.mat')

% Number of rows:
n_row = (boston,1);

% Each row of the "Boston" dataset represents a record. 
% As requested from the Exercise,the training set needs tob 2/3 of the data base. 
% Hence the training set number is: 
% "2/3*number of records(506)=337.3"
% This needs to be rounded down as number of training sets needs to be an integer.
% The function 'floor' is used to to round down the value of the result:
n_train_set = floor(2/3.*n_row); 

% "set_x" represents the x values and as requested from the question,
% It is a vector of ones.
set_x = ones(n_row,1);

% Linear regression is used to predict with mean of the y-value from training set.  
% w is a vector which is worked out as follows:
% w = (x'*x) \ (x'*y) 
% Since x_set is 1, expression (x'*x) will be the number of elements within
% vector of x which is 506. 
% Furthermore,(x'*y) gives [y1+y2+..+y506] as each element in x is 1.
% Therefore, learned w is merely same as calculating the mean. 
set_y = boston(:,14);

% At each iteration, at random, 2/3 of the data points is chosen for the
% training set. 
% The following for loop demonstrates the prediction of y which as
% explained is the same as the learned w. This is then itrated 20 times.

MSE_train_part_a=zeros(1,20);
MSE_test_part_a=zeros(1,20);

for i = 1:20
    
    % The data_sample function is built to get a random permuation from
    % the data base.
    % The function randperm(n_row) returns vector of "ones" to the shuffled
    % n_row.  
    data_sample = randperm(n_row);
    
    % These elements are then used to as indices for "set_x" on the training set.
    % The function below returns an array where "set_x" is its elements.  
    x_train = set_x(data_sample(1:n_train_set),:); 
    
    % This procedure is repeated on the rest of the database for "x_test".    
    x_test = set_x(data_sample(n_train_set+1:end),:);
    
    % Similar prodecure is done for "y_train" and "y_test".     
    y_train = set_y(data_sample(1:n_train_set),:);
    y_test = set_y(data_sample(n_train_set+1:end),:);
    
    % As the name suggests w_learned is the learned w vector and it is used 
    % The function w_calc is explained  in details in its own file.
    % It essensially calculates w using (X'X)\(X'y).
    w_learned_a(i) = w_calc(x_train,y_train);
    
    % MSE_train_part_a and test_MSE_part_a are the mean square error
    % for train and test sets.
    MSE_train_part_a(i) = MSE(x_train,y_train,w_learned_a(i)); 
    MSE_test_part_a(i) = MSE(x_test,y_test,w_learned_a(i));
end

% The avarage value for each array is then taken.
avg_train_MSE_part_a = mean(MSE_train_part_a);
avg_test_MSE_part_a = mean(MSE_test_part_a);



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %                       
                      
% Supervised Learning 1) Section 1) Exercise 9) Part b)


%  For each iteration and aeach of the 13 attributes regression is 
%  performed and MSE is calculated.
%  To operate this, 2 for loops are created  one to
%   loop through the attributes (which are columns in the boston matrix),
%   and the other loop for the simulations. 

% Number of columns:

n_col =(boston,2); 

% The bias is merely a colomn vector of 1.
bias = ones(n_row,1); 

MSE_train_part_b=zeros(1,20);
MSE_test_part_b=zeros(1,20);

% This for loops goes through the attributes 
for i = 1:n_col-1  
    
   % This for loop iterates:
   for k = 1:20 
       % set_xb will be the entire column for each attribute.
       set_xb = boston(:,i);
       % set_y stays the same and is the last column .      
       set_y = boston(:,14);
       % x and y values fro training/test sets are generated randomly.        
       x_train_b = [set_xb(data_sample(1:n_train_set),:) bias(1:n_train_set)];
       x_test_b = [set_xb(data_sample(n_train_set+1:end),:) bias(n_train_set+1:end)];
       y_train_b = set_y(data_sample(1:n_train_set),:);
       y_test_b = set_y(data_sample(n_train_set+1:end),:);
       % learned value is calclated using w_calc.      
       w_learned_b = w_calc(x_train_b,y_train_b);
       
       % MSE is then taken:      
       MSE_train_part_b(i,k) = MSE(x_train_b,y_train_b,w_learned_b);
       MSE_test_part_b(i,k) = MSE(x_test_b,y_test_b,w_learned_b);
   end
   
   % for each iteration the avarage MSE is caluclated    
   avg_train_MSE_part_b(i) = mean(MSE_train_part_b(i,:));
   avg_test_MSE_part_b(i) = mean(MSE_test_part_b(i,:));
end

   


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% Supervised Learning 1) Section 1) Exercise 9) Part c)


set_y = boston(:,14);
for i = 1:20

    x_train_c = set_x_c(data_sample(1:n_train_set),:);
    x_test_c = set_x_c(data_sample(n_train_set+1:end),:);
    y_train_c = set_y(data_sample(1:n_train_set),:);
    y_test_c = set_y(data_sample(n_train_set+1:end),:);
    w_learned_c = w_calc(x_train_c,y_train_c);
    MSE_train_part_c(i) = MSE(x_train_c,y_train_c,w_learned_c);
    MSE_test_part_c(i) = MSE(x_test_c,y_test_c,w_learned_c);
end
avg_train_MSE_part_c = mean(MSE_train_part_c);
avg_test_MSE_part_c = mean(MSE_test_part_c);




