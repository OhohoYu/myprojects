clear 

X=[1,2,3,4];
Y=[3,2,0,5];

p1 = polynomialfit(X,Y,1);
p2 = polynomialfit(X,Y,2);
p3 = polynomialfit(X,Y,3);
p4 = polynomialfit(X,Y,4);

x = 1:4;
y1 = polynomialeval(p1,x);
y2 = polynomialeval(p2,x);
y3 = polynomialeval(p3,x);
y4 = polynomialeval(p4,x);

e1 = mean_square_error(y1,Y)
e2 = mean_square_error(y2,Y)
e3 = mean_square_error(y3,Y)
e4 = mean_square_error(y4,Y)