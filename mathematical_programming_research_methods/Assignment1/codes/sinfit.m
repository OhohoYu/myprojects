function [s A] = sinfit( x, y, k )

for i = 1:k
    A(:,i) = sin(i*pi*x);
end
s = A\y;


