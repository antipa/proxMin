function out = eight_bit(in)
t = (in>=0).*(in<=255);
out = double(in.*t);