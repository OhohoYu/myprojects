clear 

X=[1,2,3,4];
Y=[3,2,0,5];

p1 = polynomialfit(X,Y,1);
p2 = polynomialfit(X,Y,2);
p3 = polynomialfit(X,Y,3);
p4 = polynomialfit(X,Y,4);

x = linspace(0,5,500);
y1 = polynomialeval(p1,x);
y2 = polynomialeval(p2,x);
y3 = polynomialeval(p3,x);
y4 = polynomialeval(p4,x);

plot(X,Y,'or')
axis([0,5,-4,8])
ax = gca;
ax.XAxisLocation = 'origin';

hold on 
plot(x,y1,'--b','LineWidth',2)
plot(x,y2, ':k','LineWidth',2)
plot(x,y3, '-.m','LineWidth',2)
plot(x,y4, '-g','LineWidth',2)
hold off

legend('Given data','k=1','k=2','k=3','k=4','Location','eastoutside')
