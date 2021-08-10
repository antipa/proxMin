function [f, g] = linear_gradient(x,A,Ah,b)
u = A(x);
r = u-b;
g = Ah(r);
f = .5*norm(r(:))^2;