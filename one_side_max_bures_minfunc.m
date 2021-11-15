function [div,alpha] = one_side_max_bures_minfunc(K,x_idx,params)
% one_side_max_bures_minfunc
% Inputs: A = D_\mu^0.5 K_XZ  and C= D_\nu^0.5 K_YZ
%           or vice versa
%       K is full kernel matrix
% Purpose: compute one side of the max sliced kernel Bures
% \omega(*) = \sum_i \alpha_i K(z_i,*) 
% Outputs: div  (objective value), alpha (parameters of the solution)
% Austin J. Brockmeier (ajbrockmeier@gmail.com)
% 8/11/2020

m = sum(x_idx); % sample size for X
n = size(K,1) - m; % sample size for Y

K = (K + K.')/2; % ensure symmetric
K_X_Z  = K(  x_idx,  :); % Z = X union Y
K_Y_Z  = K( ~x_idx, :);
% K_X_Z  = A;
% K_Y_Z  = B;

A = sqrt(1/n)*K_Y_Z; % D_\nu^\frac{1}{2}  K_{YZ}
B = sqrt(1/m)*K_X_Z; % D_\mu^\frac{1}{2}  K_{XZ}

N = size(K,1);

nMonte = 2; % number of random initializations
max_iter = N^2;
stop_delta = 1e-6;
verbose = false;
if nargin==4
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

g = @(alpha) sqrt(sum((A*alpha).^2));
h = @(alpha) sqrt(sum((B*alpha).^2));

obj = @(x) h(x) - g(x); %to Maximize
% or technically  ( h(x) - h(x) ) / (x'K x)  
% or equivalently \min_x g(x) - h(x)


objs = nan(nMonte,1); % to collect objective across different
% initial starting points

for monte_ii = 1:nMonte
    % INITIALIZATION FOR PRIMAL
    if monte_ii == 1
        %    alpha_k = rand(N,1); % random
        alpha_k = sign(sum(B,1)-sum(A,1))';
    else
        alpha_k = randn(N,1); % random
    end
    % ensure that it is feasible
    alpha_k = alpha_k / sqrt(alpha_k'*K*alpha_k);
    
    options=[];
    %options.Corr=300;
    options.Display='off';
    % END INITIALIZATION
    % MAIN LOOP
    x = 0.98*alpha_k;
    t = 100;
    for k = 0:max_iter
        %       [f,grad] = DC_barrier(alpha_k,A,C,K,t);
        %        x = fminunc(@(x) DC_barrier(x,A,C,K,t),0.98*alpha_k,opts);
        costfungrad = @(x) DC_barrier(x,A,B,K,t);
        %             fastDerivativeCheck(costfungrad,0.98*alpha_k(:),1,2)
        %             derivativeCheck(costfungrad,0.98*alpha_k(:),1,1)
        [x,~]=minFunc(costfungrad,x,options);
        alpha_k_1 = x; % new primal
        alpha_k_1 = alpha_k_1 / sqrt(alpha_k_1'*K*alpha_k_1);
        obj_val = obj(alpha_k_1); % compute current objective
        
        if k>0 && obj_val<old_obj
            if verbose
                %fprintf('Stop: Objective not increasing\n\n')
            end
            break;
        end        
        diff = abs(alpha_k - alpha_k_1);
        stop_crit = max( min( [diff, diff./abs(alpha_k) ]));
        if verbose
            %fprintf('%2d: t = %.2f del=%.7f obj=%.7f \n',k, t, norm(diff),obj_val)
        end
        if stop_crit < stop_delta
            if verbose
                %fprintf('Stop: stop_crit=%0.7f < %0.7f \n', stop_crit,stop_delta)
            end
            break;
        end
        alpha_k = alpha_k_1;
        old_obj = obj_val;
        t = t/2;
    end
    objs(monte_ii)=obj_val;
    if monte_ii == 1
        alpha = alpha_k_1;
    elseif objs(monte_ii)>max(objs(1:monte_ii-1))
        alpha = alpha_k_1;
    end
    
end
div = max(objs);
%if verbose
    %fprintf('Obj= %.7f \n\n', div)
%end
%%
% if  nargin> 2 && tests > 1
%     p1=zeros(tests,1);
%     parfor t=1:tests
%         new_idx = x_idx(randperm(n+m));
%         [p1(t)] = one_side_max_bures_minfunc(K,new_idx);
%     end
%     if nargout <= 6
%         divs = mean(p1>divs); % proportion of the shuffle less than D_22
%         %D2 = 1 - mean(D2>p2);
%     end
% end
    
end
%%

function [f,grad] = DC_barrier(x,A,C,K,t)
% Difference of convex with barrier with
% quadratic constraint
g = @(alpha) sqrt(sum((A*alpha).^2));
h = @(alpha) sqrt(sum((C*alpha).^2));
obj = @(x) g(x) - h(x); %to minimize

c = 1-x'*K*x;
psi = -log(c);
f = obj(x) + t*psi;
g_d =  (A'*(A*x))/sqrt(eps + x'*(A'*(A*x)));
h_d =  (C'*(C*x))/sqrt(x'*(C'*(C*x)));

psi_d = 2*(K*x)/c;

grad = (g_d - h_d) + t*psi_d;
end



