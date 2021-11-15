function [div,w] = one_side_max_bures_adam(A,B,max_iter)
% one_side_max_bures_adam
% Inputs: A and B are uncentered second-moment matrices (A = 1/m XX', ...)
% Purpose: compute one side of the max sliced frechet distance
% Outputs: div  (objective value), w (parameters of the solution)

if nargin<3
    max_iter = 50;
end

d = size(B,1);

assert(d == size(B,2),'First matrix must be square.')
assert(d == size(A,1),'Must be same dimension.')
assert(d == size(A,2),'Second matrix must be square.')


nMonte = 1; % number of random initializations
stop_delta = 1e-6; % value or variable
verbose = 1;

epsilon1 = 0;
g = @(w) sqrt(max(epsilon1,  w'*B*w));
h = @(w) sqrt(max(epsilon1,  w'*A*w));
obj = @(x) (h(x)-g(x))/sqrt(epsilon1 + x'*x);%to Maximize


% or technically  ( h(x) - h(x) ) / (x'*x)
% or equivalently \min_x g(x) - h(x)
obj_val = -inf;
% initial starting points
w_k = randn(d,1);
% ensure that it is feasible
w_k = w_k / sqrt(w_k'*w_k);
% ADAM
lr = 1e-3;
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-08;

m = 0*w_k;
v = 0*w_k;

for iter = 1 : max_iter
    [~,grad] = DC_grad(w_k,A,B,epsilon1);
    m = beta1*m + (1-beta1)*grad; % AR model for grad
    v = beta2*v + (1-beta2)*grad.^2; % AR model for grad squared
    alpha_t = lr*sqrt(1-beta2^(iter))/(1-beta1^(iter));
    w_k_1 = w_k; % save the old weights
    w_k = w_k - alpha_t*m./(sqrt(v)+epsilon); % update the weights
    
    obj_val = obj(w_k/sqrt(w_k'*w_k));
    
    diff = abs(w_k - w_k_1);
    stop_crit = max(diff); 

    if verbose
        %fprintf('%2d:  del=%.7f obj=%.7f \n',iter, norm(diff),obj_val)
    end
    if stop_crit < stop_delta
        if verbose
            fprintf('Stop: stop_crit=%0.7f < %0.7f \n', stop_crit,stop_delta)
        end
        break;
    end
end
w = w_k / sqrt(w_k'*w_k);
div  = obj(w);
if verbose
    %fprintf('Obj= %.7f \n\n', div)
end

end


function [f,grad] = DC_grad(x,A,B,epsilon1)
% Difference of convex
%g = @(w) sqrt(0.01 + w'*B*w);
%h = @(w) sqrt(w'*A*w);
%f = @(w) (g(w) - h(w))/sqrt(w'*w); %to minimize
%obj = @(w) h(w) - g(w)

Bx = B*x;
Ax = A*x;
xx = x'*x;
xBx = x'*Bx;
xAx = x'*Ax;

g_x = sqrt(max(0,epsilon1+xBx));
h_x = sqrt(max(0,epsilon1+xAx));
f = (g_x-h_x)/sqrt(epsilon1+xx);

g_d =  B*x/g_x;
h_d =  (Ax)/h_x;
grad = (g_d - h_d)/sqrt(xx) - x*(g_x-h_x)/xx;
end

