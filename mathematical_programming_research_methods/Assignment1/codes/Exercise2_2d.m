clear 
st_dev = 0.07; % standard deviation
poly_no = 18;  % number of polynomial fits
sample_no_train = 30; % number of samples from [0,1] for train set
sample_no_test = 1000; % number of samples from [0,1] for test set

% for train set
X_train = linspace(0,1,sample_no_train); % sample uniformly
Y_train = gfunction(X_train,st_dev); % g_0.07 applied to each element of X_train
P_train = zeros(poly_no, sample_no_train); % matrix storing polynomial train coords
E_train = zeros(1, poly_no); % array containing MSE for each train fit
% for test set
X_test = linspace(0,1,sample_no_test); % sample uniformly
Y_test = gfunction(X_test,st_dev); % g_0.07 applied to each element of X_test
P_test = zeros(poly_no, sample_no_test); % matrix storing polynomial test coords
E_test = zeros(1, poly_no); % array containing MSE for each test fit

i=1; 
% loop to store coefficients for each fit in a cell array coeffs_train
while(i<poly_no+1)
  coeffs_train{i} = polynomialfit(X_train,Y_train,i); % poly coefficients stored here
  P_train(i,:) = polynomialeval(coeffs_train{i}, X_train); % P_train row update 
  P_test(i,:) = polynomialeval(coeffs_train{i}, X_test); % P_train row update 
  E_train(i) = mean_square_error(P_train(i,:),Y_train); % train error
  E_test(i) = mean_square_error(P_test(i,:),Y_test); % test error
  i = i+1;
end

% plot
basis_dim = 1:poly_no;
plot(basis_dim, log(E_test), 'r', 'Linewidth', 1.5) % plot dim 
ax = gca;
ax.XAxisLocation = 'origin';