function [out, norm_out] = soft_wavelet_2d(x,wavelev,wavetype,tau,minval,maxval,pad)
%Inputs:
%x: variable
%wavelev: wavelet level for wavedec
%wavetype: wavelet type
%tau: soft threshold amount
%minval: minimum allowed value in final output
%maxval: same but max
%W: windowing function. Optional, but needed for diffuser 2d problem. Set
%to ones if not using.
[C,S] = wavedec2(x,wavelev,wavetype);
C_soft = soft(C,tau);
norm_out = tau*norm(C,1);
out = pad(min(max(waverec2(C_soft,S,wavetype),minval),maxval));
