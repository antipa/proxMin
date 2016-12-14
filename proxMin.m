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
%h1 = figure(1);
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
        tk = 1;
        xk = x0;
        yk = xk;
        f = 1e12;
        f_kp1 = f;
        tic
        while (step_num < options.maxIter) && (f>options.residTol)
            
            step_num = step_num+1;
            [f_kp1, g] = GradErrHandle(yk);
            [x_kp1, norm_x] = ProxFunc(yk-options.stepsize*g);
            t_kp1 = (1+sqrt(1+4*tk^2))/2;
            beta_kp1 = (tk-1)/t_kp1;
            dx = x_kp1-xk;
            y_kp1 = x_kp1+beta_kp1*(dx);
            restart = (yk(:)-x_kp1(:))'*dx(:);
            if ~mod(step_num,options.disp_fig_interval)
                if options.disp_figs
                    draw_figures(yk,options);
                end
                toc
                if options.known_input
                    fprintf('%i\t %6.4e\t %6.4e\t %.3f\t %6.4e\t %.2f dB\n',...
                        step_num,f,norm_x,tk,...
                        sum(sum((options.crop(options.xin)-options.crop(yk)).^2))/numel(options.crop(yk)),...
                        psnr(options.crop(gather(yk)),options.crop(options.xin),255));
                else
                    fprintf('%i\t%6.4e\t%6.4e\t%.3f\n',step_num,f,norm_x,tk)
                end
                tic
            end
            if restart>0
                tk = 1;
            else
                tk = t_kp1;
            end
            
            xk = x_kp1;
            yk = y_kp1;
            f = f_kp1;
            
            
            if abs(restart)<options.convTol
                fprintf('Answer is stable to within convTol. Stopping.\n')
                out = yk;
                draw_figures(out,options);
                break
            end
            
            
            
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
set(0,'CurrentFigure',options.fighandle)
if numel(options.xsize)==2
    imagesc(options.disp_crop(xk))
    axis image
    colorbar
    colormap parula
    caxis(gather([prctile(xk(:),.1) prctile(xk(:),99.9)]))
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

