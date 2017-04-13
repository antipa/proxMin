function [f, g] = matrixError(A,At,x,b)
rv = A*x(:)-b;
f = norm(rv)^2;
g = At*rv;