function result = gfunction(x,sigma)
% This function takes as input an array x and a standard deviation value
% sigma. sin^2(2*pi*x) is computed for each element of the array. Normal 
% noise is added afterwards and the array result is outputted.
i = 1;
% noise values are selected from a normal distribution of mean 0, sd sigma.
noise = randomnorm(0,sigma,length(x)); 
% loop to compute resulting elements, updated with their respective noise.
while(i<length(x)+1)
  result(i) = (sin(2*pi*x(i)))^2 + noise(i);
  i = i+1;
end
end

 