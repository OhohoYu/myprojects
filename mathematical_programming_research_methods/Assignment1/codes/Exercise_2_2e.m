clear 

st_dev = 0.07; % standard deviation
poly_no = 18;  % number of polynomial fits
sample_no_train = 30; % number of samples from [0,1] for train set 
sample_no_test = 1000; % number of samples from [0,1] for test set 
runs = 100; % 100 runs 

% for train set 
X_train = linspace(0,1, sample_no_train); % sample uniformly
P_train = zeros(poly_no, sample_no_train); % matrix storing polynomial train coords
E_train = zeros(1, poly_no); % array containing MSE for each train fit
E_tot_train = zeros(1, poly_no); % total err for each train fit for avg calculation

% for test set
X_test = linspace(0,1, sample_no_test); % sample uniformly
P_test = zeros(poly_no, sample_no_test); % matrix storing polynomial train coords
E_test = zeros(1, poly_no); % array containing MSE for each train fit
E_tot_test = zeros(1, poly_no); % total err for each test fit for avg calculation

m = 0;
while (m<runs)
    Y_train = gfunction(X_train, st_dev); % g_0.07 applied to each element of X_train
    Y_test = gfunction(X_test, st_dev); % g_0.07 applied to each element of X_test
    i=1; 
    while(i<poly_no+1)
        coeffs_train{i} = polynomialfit(X_train,Y_train,i); 
        P_train(i,:) = polynomialeval(coeffs_train{i}, X_train); % P_train row update 
        P_test(i,:) = polynomialeval(coeffs_train{i}, X_test); % P_test row update
        E_train(i) = mean_square_error(P_train(i,:),Y_train); % train error
        E_test(i) = mean_square_error(P_test(i,:),Y_test); % test error
        E_tot_train(i) = E_tot_train(i) + E_train(i); % total train err for avg
        E_tot_test(i) = E_tot_test(i) + E_test(i); % total test err for avg
        i = i+1;
    end
    m = m+1;
end

E_avg_train = E_tot_train/runs;
E_avg_test = E_tot_test/runs;

% plot
basis_dim = 1:poly_no;
plot(basis_dim, log(E_avg_train), 'r', 'Linewidth', 1.5) % plot dim 
ax = gca;
ax.XAxisLocation = 'origin';
hold on 
plot(basis_dim, log(E_avg_test), '--b', 'Linewidth', 1.5) % plot dim 
hold off
legend('ln te_k(S)','ln tse_k(S,T)','Location','eastoutside')