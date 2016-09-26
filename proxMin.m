function [out,varargout] = proxMin(GradErrHandle,ProxFunc,x0,b,options)

% Out = WavelengProxGrad(GradErrHanle,ProxHandle,AxyTxy0,options)
%
% GradErrHandle: handle for function that computes error and gradient at
%   each step
%
% ProxFunc: handle for function that does projection step
%
% AxyTxy0: initialization Nx x Ny x 2
%   where the first matrix is the amplitude of the diffuser A(x,y)
%   and the second matrix is the thickness of the diffuser T(x,y)
%
% options: similar to minFunc, but won't support all of the same options.
%
% Nick Antipa, summer 2016
if ~isa(GradErrHandle,'function_handle')
    GradErrHandle = @(x) matrixError(GradErrHandle,transpose(GradErrHandle),x,b);
end
if ~isfield(options,'convTol')
    options.convTol = 1e-9;
end
if ~isfield(options,'residTol')
    options.residTol = 1e-2;
end
if ~isfield(options,'xsize')
    options.xsize = size(A,2);
end
if ~isfield(options,'momentum')
    options.momentum = 'linear';
end
if ~isfield(options,'disp_figs')
    options.disp_figs = 0;
end
step_num = 0;
yk = x0;
h1 = figure(1);
fprintf('Iteration\t objective\t ||x||\tmomentum\n');
fun_val = zeros(options.maxIter,1);
%step_size = .0000000008;
step_size = options.stepsize;
fm1 = 0;
f = inf;
switch lower(options.momentum)
    case('linear')
        while (step_num < options.maxIter) && (f>options.residTol);
            
            step_num = step_num+1;
            [ f, g ] = GradErrHandle( yk );
            fun_val(step_num) = f;
            x_t1 = yk - step_size*g;
            yk = ProxFunc(x_t1);
            
            if ~mod(step_num,options.disp_fig_interval)
                if options.disp_figs
                    draw_figures(yk,options)
                end
            end
            
            if abs(fm1-f)<options.convTol
                fprintf('Answer is stable to within convTol. Stopping.\n')
                out = yk;
                break
            end
            
            fm1 = f;
            
            
            fprintf('%i\t%6.4e\n',step_num,f)
        end
        
    case ('nesterov')
        t_km1 = 1;
        x_km1 = x0;
        y_km1 = x_km1;
        fm1 = inf;
        norm_x_km1 = inf;
        while (step_num < options.maxIter) && (f>options.residTol);
            
            step_num = step_num+1;
            [f, g] = GradErrHandle(y_km1);
            fun_val(step_num) = f;
            uk = y_km1 - step_size*g;
            [xk, norm_x] = ProxFunc(uk);
            f = f+norm_x_km1;
            if f-fm1>0
                %fprintf('Resetting momentum\n')
                tk = 1;
                t_km1 = 1;
                xk = x_km1;
            else
                tk = (1+sqrt(1+4*t_km1^2))/2;
            end
            %xk = xk;
            yk = xk+(t_km1-1)/tk*(xk-x_km1);
            if ~mod(step_num,options.disp_fig_interval)
                if options.disp_figs
                    draw_figures(yk,options);
                end
                if options.known_input
                    fprintf('%i\t %6.4e\t %6.4e\t %.3f\t %6.4e\t %.2f dB\n',...
                        step_num,f,norm_x,tk,...
                        sum(sum((options.crop(options.xin)-options.crop(yk)).^2))/numel(options.crop(yk)),...
                        psnr(options.crop(yk),options.crop(options.xin),255));
                else
                    fprintf('%i\t%6.4e\t%6.4e\t%.3f\n',step_num,f,norm_x,tk)
                end
            end
            
            if abs(fm1-f)<options.convTol
                fprintf('Answer is stable to within convTol. Stopping.\n')
                out = y_km1;
                draw_figures(out,options);
                break
            end
            
            
            fm1 = f;
            y_km1 = yk;
            t_km1 = tk;
            x_km1 = xk;
            norm_x_km1 = norm_x;
            
        end
        
end
if (f<options.residTol)
    fprintf('Residual below residTol. Stopping. \n')
end
if step_num>=options.maxIter
    fprintf('Reached max number of iterations. Stopping. \n');
end
out = yk;
if nargout>1
    varargout{1} = fun_val;
end
draw_figures(out,options)
return

function draw_figures(xk,options)
if numel(options.xsize)==2
    imagesc(options.disp_crop(reshape(xk,options.xsize)))
    axis image
    colorbar
    colormap parula
    caxis([prctile(xk(:),.1) prctile(xk(:),99.9)])
elseif numel(options.xsize) == 4
    xkr = reshape(xk,options.xsize);
    subplot(2,2,1)
    imagesc(transpose(squeeze(xkr(end,ceil(options.xsize(2)/2),:,:))))
    hold on
    axis image
    colorbar
    colormap gray
    caxis([0 prctile(xkr(:),99)]);
    hold off
    
    subplot(2,2,2)
    imagesc(transpose(squeeze(xkr(1,ceil(options.xsize(2)/2),:,:))))
    hold on
    axis image
    colorbar
    colormap gray
    caxis([0 prctile(xkr(:),99)]);
    hold off
    
    subplot(2,2,3)
    imagesc(transpose(squeeze(xkr(ceil(options.xsize(2)/2),1,:,:))))
    hold on
    axis image
    colorbar
    colormap gray
    caxis([0 prctile(xkr(:),99)]);
    hold off
    
    subplot(2,2,4)
    imagesc(transpose(squeeze(xkr(ceil(options.xsize(2)/2),end,:,:))))
    hold on
    axis image
    colorbar
    colormap gray
    caxis([0 prctile(xkr(:),99)]);
    hold off
    
    
elseif numel(options.xsize)==1
    
    plot(xk)
end
drawnow

