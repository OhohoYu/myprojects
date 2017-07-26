clear

[X,Y] =  meshgrid(linspace(0,5,15), linspace(0,5,15));
mesh(X,Y,fcarg(X,Y));
xlabel('x')
ylabel('y')
zlabel('f(x,y)')