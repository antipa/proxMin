function [f, g] = matrixError(A,At,x,b)
rv = A*x(:)-b;
f = norm(rv);
g = At*rv;