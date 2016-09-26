function [f, g] = matrixErrorv2(A,AtA,x,Atb,b)
rv = A*x(:)-b;
f = norm(rv);
g = AtA*x-Atb;