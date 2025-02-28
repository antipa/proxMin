function [out, norm_out] = soft_wavelet_2d(x,wavelev,wavetype,tau,cycle,nonneg,thresh_type)
%[out, norm_out] = soft_wavelet_2d(x,wavelev,wavetype,tau,cycle,nonneg,thresh_type)
%Inputs:
%x: variable
%wavelev: wavelet level for wavedec
%wavetype: wavelet type
%tau: soft threshold amount
%minval: minimum allowed value in final output
%maxval: same but max
%W: windowing function. Optional, but needed for diffuser 2d problem. Set
%to ones if not using.

if cycle
    Ndims = numel(size(x));
    Nshift = 2;
else
    Ndims = 1;
    Nshift = 1;
end
out = zeros(size(x),'like',x);
norm_out = 0;
for n = 1:Ndims
    for m = 1:Nshift
        [C,S] = wavedec2(circshift(x,m-1,n),wavelev,wavetype);
        %C_soft = C;
        C_soft = C;
        for k = 1:wavelev
            %C_soft = soft_wave_detail('c',C_soft,S,k,tau*sqrt(k));
            C_soft = wthcoef2('h',C_soft,S,k,tau*sqrt(k),thresh_type);
            C_soft = wthcoef2('v',C_soft,S,k,tau*sqrt(k),thresh_type);
            C_soft = wthcoef2('d',C_soft,S,k,tau*sqrt(k),thresh_type);
            %norm_out = norm_out + tau*norm(detcoef2('c',C_soft,S,denoise_level(n)),1);
        end
        %C_soft = soft(C,tau);
        norm_out = norm_out + 1/(Nshift*Ndims)*tau*norm(C_soft,1);

        out = out + 1/(Nshift*Ndims)*circshift(waverec2(C_soft,S,wavetype),-m+1,n);
    end
end
if nonneg
    out = 0.5*out + 0.5*max(x,0);
end    

return
