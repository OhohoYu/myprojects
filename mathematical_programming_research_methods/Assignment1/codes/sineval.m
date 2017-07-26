function Y = sineval(S,X) 
    size_s = size(S);   % coefficient size
    size_x = size(X);   % size of input x-coords
    Y = zeros(size(X));   % initialise matrix for predicted Y
    for i =1:size_x  % loop over rows
        for j=1:size_s % loop over columns
            Y(1,i) = Y(1, i) + S(j,1)*sin(j*pi*x(i));
        end
    end    
end