clear 

sample_no = 30; % number of samples from [0,1]
poly_no = 18;  % number of polynomial fits
st_dev = 0.07; % standard deviation

% for noisy dataset plot
X = linspace(0,1,sample_no); % sample uniformly
Y = gfunction(X,st_dev); % g_0.07 applied to each element of X1
P = zeros(poly_no, sample_no); % matrix storing y-coords of polynomial fits
E = zeros(1, poly_no); % array containing MSE for each fit

i=1; 
% loop to store coefficients for each polynomial fit in a cell array coeffs
while(i<poly_no+1)
  coeffs{i} = polynomialfit(X,Y,i); % coefficients of each fit stored here
  P(i,:) = polynomialeval(coeffs{i}, X); % P row (each row is a different fit) update 
  E(i) = mean_square_error(P(i,:),Y); % errors stored here using immse function
  i = i+1;
end

% plot
basis_dim = 1:poly_no;
plot(basis_dim, log(E), 'r', 'Linewidth', 1.5) % plot dim 
ax = gca;
ax.XAxisLocation = 'origin';





