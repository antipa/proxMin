function y = TVnorm(x)
dh = cat(2,x(:,1)-x(:,end),diff(x,1,2));
dv = cat(1,x(1,:)-x(end,:),diff(x,1,1));
y = sum(sum(sqrt(dv.^2+dh.^2)));
