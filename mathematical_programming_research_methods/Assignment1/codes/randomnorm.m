function numbers = randomnorm(mu, sigma, n)
% randomnorm.m returns a size n array of random numbers selected from a 
% normal distribution with standard deviation sigma and mean mu. 
i = 1;      
while(i<n+1)   % while loop over the array setting random elements
  % rescaling operation
  numbers(i) = randn()* sigma + mu;
  i = i+1;
end    
end


