function [div,alpha] = max_sliced_kernel_wasserstein_minfunc(K,x_idx,params)
% max_sliced_kernel_wasserstein_minfunc
% Inputs: K is kernel matrix, x_idx is logical index for one samples
% Purpose: compute the max-sliced kernel Wasserstein-2 metric
% \omega(*) = \sum_i \alpha_i K(z_i,*) 
% Outputs: div  (objective value), alpha (parameters of the solution)
% Austin J. Brockmeier (ajbrockmeier@gmail.com)
% 8/11/2020

N = size(K,1);


nMonte = 20; % number of random initializations
if nargin==4
    if isfield(params,'nMonte') 
        nMonte = params.nMonte;
    end
end

m = sum(x_idx);
n = size(K,1) - m;
K = (K + K.')/2;
K_X_Z  = K(  x_idx, :);
K_Y_Z  = K( ~x_idx, :);


A = K_X_Z;
B = K_Y_Z;
R = 1/m*(K_X_Z'*K_X_Z) + 1/n*(K_Y_Z'*K_Y_Z);

h = @(alpha) alpha'*R*alpha...
    -2*(alpha'*K_X_Z')*sortOT(K_X_Z*alpha, K_Y_Z*alpha)*(K_Y_Z*alpha);
%h2 = @(alpha) trace(sortOT(K_X_Z*alpha, K_Y_Z*alpha)'*pdist2(K_X_Z*alpha, K_Y_Z*alpha,'squaredeuclidean'));
% h==h2

g = @(alpha) alpha'*K*alpha; 
obj = @(x) h(x)/g(x); %to Maximize

objs = nan(nMonte,1); % to collect objective across different
% initial starting points
for monte_ii = 1:nMonte
    % INITIALIZATION FOR PRIMAL
    if monte_ii == 1
            Q = mean(K_X_Z,1)'*mean(K_Y_Z,1);
            Q = R - Q - Q';
            [alpha_k,~] =  eigs(Q+Q', K+1e-9*eye(m+n), 1);
    else
        alpha_k = randn(N,1); % random
    end 
    % ensure that it is feasible
    alpha_k = alpha_k / sqrt(alpha_k'*K*alpha_k);
    
    options=[];
    options.Display='off';
    % END INITIALIZATION
    % MAIN LOOP
    costfungrad = @(x) do_unconstrained(x,K,R,A,B);
    [x,~] = minFunc(costfungrad,alpha_k,options);
    x = x / sqrt(x'*K*x);
    obj_val = obj(x); % compute current objective
    alpha_k_1 = x;
        
    objs(monte_ii)=obj_val;
    if monte_ii == 1
        alpha = alpha_k_1;
    elseif objs(monte_ii)>max(objs(1:monte_ii-1))
        alpha = alpha_k_1;
    end
    
end

div = sqrt(max(objs));
%fprintf('Obj= %.7f \n\n', div)
%%
% if  nargin> 2 && tests > 1
%     p1=zeros(tests,1);
%     parfor t=1:tests
%         new_idx = x_idx(randperm(n+m));
%         [p1(t)] = max_sliced_kernel_wasserstein_minfunc(K,new_idx);
%     end
%     if nargout <= 6
%         divs = mean(p1>divs); % proportion of the shuffle less than D_22
%         %D2 = 1 - mean(D2>p2);
%     end
% end

    
end
   

function [f,grad] = do_unconstrained(x,K,R,A,B)

P = sortOT( A*x, B*x);
% quadratic step
Q = A'*P*B;
C = R - Q - Q';
n1 = x'*C*x;
n2 = x'*K*x;
Grad = 2/n2*(C*x) - 2*n1/n2^2*(K*x);
f = -n1/n2;
grad = -Grad;
end    
    