% Loading Boston dataset

clear all; 
close all; 
clc;

data = load('boston.mat');
sample = data.boston;


% Initialsing the variables gamma and sigma as instructed from the question 

a=[-40:-26];
length_gamma=length(a);
gamma=(2*ones(1,(length_gamma))).^a;


b=[7:0.5:13];
length_sigma=length(b);
sigma=(2*ones(1,(length_sigma))).^b;

% The total number of the rows in the data base is needed for the splitting
% of train,valdation and test sets. The training set is 2/3 of the total
% dataset.

[rows,~]=size(sample);
train_set=floor(2/3*rows);

% Initialising number of K foldds and iterations.
k_folds=5;
iteration=20;
fold_subset=round(train_set/k_folds);

% These zero arrays elemsnts will get filled after each iterations.

min_mse_val=zeros(1,iteration);
mse_train_final=zeros(1,iteration);



for count=1:iteration
    count
    
  % The data set is split into to main training and test set.    
    sample_data = sample(randperm(rows),:);
    
    set_x =  sample_data(:,1:13);
    set_y =  sample_data(:,14);
    
    x_train = set_x(1:train_set,:);
    y_train = set_y(1:train_set,:);
    
    x_test = set_x(train_set + 1:rows,:); 
    y_test = set_y(train_set + 1:rows,:);
    
   % The following for loop attempts to separate 5 different layers within
   % the main training set. At each iteration one of five the subsets 
   % will become the validation set and the remaining four sets of the main
   % training set will become the sub training set. The validation layer will 
   % change and shuffle each time with the help of function circshift.
   
      for k = 1:k_folds
            x_val_fold = x_train(1:fold_subset,:);
            x_train_fold = x_train(fold_subset+1:end,:);
            y_val_fold = y_train(1:fold_subset);
            y_train_fold = y_train(fold_subset+1:end);
            x_train = circshift(x_train,[fold_subset, 0]); 
            y_train = circshift(y_train,[fold_subset, 0]);
            
%             xtrain = x_train_fold(:,:,k);
      
      
%       The aim if the following for loop is to predict the value of y on
%       the validation set. 

       for i=1:length_gamma
          
          for j=1:length_sigma 
              
              kernel_matrix= Kernel_m(x_train_fold,x_val_fold,sigma(1,j));
              
              k_matrix_train=kernel_matrix(1:size(...
                  x_train_fold,1),1:size(x_train_fold,1));
              
              end_kernel=size(kernel_matrix,1);
              
              k_matrix_val=kernel_matrix(...
                  size(x_train_fold,1)+1:end_kernel,1:size(x_train_fold,1));
              
             alpha_dual=kridgereg(k_matrix_train,y_train_fold,gamma(1,i));
             mse_y_val(i,j,k)=dualcost(k_matrix_val,y_val_fold,alpha_dual);
             
        
          end
        end 
      end 
      

      % Here different validation sets are added up together which gives
      % the matrix a third dimention with the size of the k fold.
       mse_validation_set=sum(mse_y_val,3);
       
      % To calculate the avarage mse validation, it needs to be divided by
      % the number of k-folds. This will make the matrix 2 dimensional
      % again.
      
       average_mse_val=mse_validation_set/k_folds;
       
%        
%         if count==20 %only plot for one iterration
%            figure;
%            mesh(log2(sigma),log2(gamma),average_mse_val);
%            xlabel('Sigma');
%            ylabel('Gamma');
%            zlabel('Average MSE');
%       
%         end
     %  Finding the optimal sigma and gamma values from the validation set
     %  and reusing that on the main test set.
     
     [row_gamma, coloumn_sigma]=find(average_mse_val==min(min(average_mse_val)));
      opt_gamma=gamma(1,row_gamma);
      opt_sig=sigma(1,coloumn_sigma);
       
      min_mse_val(count)=min(min(average_mse_val));
      
    % re-calculating the K-matrix in order to perdict y on the test set.
     
    k_mat_train=Kernel_m(x_train,[],opt_sig);
    new_alpha=kridgereg(k_mat_train,y_train,opt_gamma);
    
    mse_train_final(count)=dualcost(k_mat_train,y_train,new_alpha);
   
  
    k_mat_test=Kernel_m(x_train,x_test,opt_sig);
    
    end_kernel_2=size(k_mat_test,1);
    
    new_alpha_2=(kridgereg((k_mat_train),y_train,opt_gamma));
    
    k_test=k_mat_test(train_set+1:end_kernel_2,1:size(x_train,1));
    
    mse_test_final(count)=dualcost(k_test,y_test,(new_alpha_2))
    
    
     
end 
    
% Part D)

mean_min_mse_train = mean(mse_train_final);
sd_min_mse_train = std(mse_train_final)

mean_min_mse_test = mean(mse_test_final)
ds_min_mse_test = std(mse_test_final)









