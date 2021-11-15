function [div,w] = one_side_max_sliced_bures_minfunc(A,B,params)
% one_side_max_bures_minfunc
% Inputs: A and B are uncentered second-moment matrices
% Purpose: compute one side of the max sliced frechet distance
% Outputs: div  (objective value), w (parameters of the solution)
% Austin J. Brockmeier (ajbrockmeier@gmail.com)
% 8/20/2020

N = size(A,1);
assert(N == size(A,2),'First matrix must be square.')
assert(N == size(B,1),'Must be same dimension.')
assert(N == size(B,2),'Second matrix must be square.')


nMonte = 2; % number of random initializations
max_iter = N^2;
stop_delta = 1e-6;
verbose = false;
if nargin==3
    if isfield(params,'verbose') 
        verbose = params.verbose;
    end
    if isfield(params,'stop_delta') 
        stop_delta = params.stop_delta;
    end
    if isfield(params,'max_iter') 
        max_iter = params.max_iter;
    end
    if isfield(params,'nMonte') 
        nMonte = params.nMonte;
    end
end

g = @(w) sqrt(w'*A*w);
h = @(w) sqrt(w'*B*w);

obj = @(x) h(x) - g(x); %to Maximize
% or technically  ( h(x) - h(x) ) / (x'*x)  
% or equivalently \min_x g(x) - h(x)


objs = nan(nMonte,1); % to collect objective across different
% initial starting points

for monte_ii = 1:nMonte
    % INITIALIZATION FOR PRIMAL
    if monte_ii == 1
        %    alpha_k = rand(N,1); % random
        w_k = sign(sum(A,1)-sum(B,1))';
    else
        w_k = randn(N,1); % random
    end
    % ensure that it is feasible
    w_k = w_k / sqrt(w_k'*w_k);
    
    options=[];
    options.Display='off';
   
    % END INITIALIZATION
    % MAIN LOOP
    x = 0.98*w_k;
    
    
    t = 100;
    for k = 0:max_iter
        costfungrad = @(x) DC_barrier(x,A,B,t);
        [x,~]=minFunc(costfungrad,x,options);%,[],[],[],[],lb,ub,[],options);
        
        w_k_1 = x; % new primal
        w_k_1 = w_k_1 / sqrt(w_k_1'*w_k_1);
        obj_val = obj(w_k_1); % compute current objective
        
        
        if k>0 && obj_val<old_obj
            if verbose
                %fprintf('Stop: Objective not increasing\n\n')
            end
            break;
        end        
        diff = abs(w_k - w_k_1);
        stop_crit = max( min( [diff, diff./abs(w_k) ]));
        %        obj_val = obj(alpha_k_1); % compute current objective
        if verbose
            %fprintf('%2d: t = %.2f del=%.7f obj=%.7f \n',k, t, norm(diff),obj_val)
        end
        if stop_crit < stop_delta
            if verbose
                %fprintf('Stop: stop_crit=%0.7f < %0.7f \n', stop_crit,stop_delta)
            end
            break;
        end
        w_k = w_k_1;
        old_obj = obj_val;
        t = t/2;
    end
    objs(monte_ii)=obj_val;
    if monte_ii == 1
        w = w_k_1;
    elseif objs(monte_ii)>max(objs(1:monte_ii-1))
        w = w_k_1;
        %    fprintf('Better: obj=%0.7f < %0.7f \n\n', max(objs(1:monte_ii-1)),objs(monte_ii))
    end
    
end
div = max(objs);
%if verbose
    %fprintf('Obj= %.7f \n\n', div)
%end
end


function [f,grad] = DC_barrier(x,A,B,t)
% Difference of convex with barrier with
% quadratic constraint
g = @(w) sqrt(w'*A*w);
h = @(w) sqrt(w'*B*w);
obj = @(x) g(x) - h(x); %to minimize

c = 1-x'*x;
psi = -log(c);
f = obj(x) + t*psi;
g_d =  (A*x)/sqrt(eps + x'*(A*x));
h_d =  (B*x)/sqrt(x'*(B*x));

psi_d = 2*x/c;

grad = (g_d - h_d) + t*psi_d;
end