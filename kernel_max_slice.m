function [omega,divs,alphas,z_idx] = kernel_max_slice(K,x_idx,method,L)
% kernel_max_slice finds a slice that maximizes a divergence
% Input: K - kernel matrix (m+n) by (m+n)
%       x_idx - logical indexing for sample of X ~ \mu
%       -->   ~x_idx - logical indexing for sample of Y ~ \nu
%       method is a string or cell dictating the method options include
%       'mmd'  'max-w2'  'max-tv'  'max-bures' 'max-frechet'
% omega is the witness function evaluations on the input for each slice
% divs is the divergence values for each slice
% alphas is the coefficients defining the witness function
% z_idx is the support points used to define the witness function

div_measure = method;
m = sum(x_idx);
n = size(K,1) - m;

% Use all points to define witness function or subsample
if nargin < 4
    L = m+n;
    z_idx = 1:m+n;
else
    if numel(L)==1
        assert(L <= m+n);
        assert(round(L)==L && L>1);
        z_idx = randperm(m+n,L);
    else
        z_idx = L;
        L = numel(z_idx);
    end
end

%Ensure symmetry for eigenvalue problems
K = (K + K.')/2;
K_X_Z  = K(  x_idx,  z_idx);
K_Y_Z  = K( ~x_idx, z_idx);
K_L = K(z_idx,z_idx);

A12 = sqrt(1/m)*K_X_Z; % D_\mu^\frac{1}{2}  K_{XZ}
B12 = sqrt(1/n)*K_Y_Z; % D_\nu^\frac{1}{2}  K_{YZ}

alphas = zeros(n+m,2);
switch lower(div_measure)
    case 'mmd'
        alphas = 1/m*x_idx - 1/n*(~x_idx);
        divs = alphas'*K*alphas;
    case {'w2','max-w2'}  %requires minFunc
        params =struct('nMonte',1);
        [div,alpha] = max_sliced_kernel_w2_minfunc(K,x_idx,z_idx,params);
        divs = div;
        alphas(z_idx,1) = alpha;
        alphas(:,2)=[];
    case {'tv','max-tv'}
        A = (A12'*A12);
        B = (B12'*B12);
        [omega,D]=eigs(A-B,K_L + 1e-9*eye(L), 2,'bothendsreal');
        alphas(z_idx,:) = omega*sign(D);
        divs = 0.5*diag(abs(D));
    case {'bures','max-bures','max-b','max-gw-u','gw-u','gw-upper'}
        switch lower(div_measure)
            case {'max-frechet', 'max-gw-u','gw-u','gw-upper'}
                A12 = sqrt(1/n)*bsxfun(@minus,K_Y_Z,mean(K_Y_Z,1));
                B12 = sqrt(1/m)*bsxfun(@minus,K_X_Z,mean(K_X_Z,1));
        end
        [div1,alpha1] = one_side_max_bures_path(A12,B12,K_L + 1e-9*eye(L));
        [div2,alpha2] = one_side_max_bures_path(B12,A12,K_L + 1e-9*eye(L));
        alphas(z_idx,:) = [alpha1,alpha2];
        divs = [div1,div2];
    otherwise
        error('unknown method  %s', div_measure)
end

switch lower(div_measure)
    case {'max-frechet', 'max-gw-u','gw-u','gw-upper'}
        %finding difference in mean
        diff = (mean(K(x_idx,:),1) - mean(K(~x_idx,:),1))';
        alpha_mmd = cat(1,1/m*ones(m,1),-1/n*ones(n,1));
        div_mmd = diff'*alpha_mmd/sqrt(alpha_mmd'*K*alpha_mmd);
        [~,idx]=max(divs);
        alpha2 = alphas(:,idx);
        alphas = [alpha_mmd,alpha2];
        divs = sqrt(div_mmd.^2 + divs.^2);
end
omega = K*alphas;
end


function [div,alpha] = one_side_max_bures_path(A12,B12,K)
% one_side_max_bures_minfunc
% Inputs: A = D_\mu^0.5 K_XZ  and B= D_\nu^0.5 K_YZ
%           or vice versa
%       K is full kernel matrix
% Purpose: compute one side of the max sliced kernel Bures
% \omega(*) = \sum_i \alpha_i K(z_i,*)
% Outputs: div  (objective value), alpha (parameters of the solution)

N = size(K,1);

g = @(alpha) sqrt(sum((B12*alpha).^2));
h = @(alpha) sqrt(sum((A12*alpha).^2));

obj = @(x) h(x) - g(x); %to Maximize
% or technically  ( h(x) - h(x) ) / (x'K x)
% or equivalently \min_x g(x) - h(x)

A = (A12'*A12);
B = (B12'*B12);

% solution in null(A)
V= null(B12);
obj_null = -inf;
if size(V,2)>0
    [beta,~]=eigs(V'*A*V,V'*K*V + 1e-9*eye(size(V,2)), 1,'largestreal');
    alpha = V*beta;
    alpha = alpha/sqrt(alpha'*K*alpha);
    obj_null  = obj(alpha);
end
K_reg = K + 1e-9*eye(N);
[gamma_star,f_gamma_star] = fminbnd(@(gamma) -get_geig_eval(gamma*A-B,K_reg,obj), 0,1);
obj_not_null = -f_gamma_star;
if obj_null > obj_not_null
    div = obj_null;
    gamma_star = inf;
    % return alpha
else
    div = obj_not_null;
    [alpha,~]=eigs(gamma_star*A-B,K + 1e-9*eye(N), 1,'largestreal');
    alpha = alpha/sqrt(alpha'*K*alpha);
end
fprintf('Obj= %.7f @ %.7f \n\n', div,gamma_star)
end

function val = get_geig_eval(A,B,obj)
[u,~]=eigs(A,B, 1,'largestreal');
alpha = u/sqrt(u'*B*u);
val = obj(alpha);
end


function [div,alpha] = max_sliced_kernel_w2_minfunc(K,x_idx,z_idx,params)
% max_sliced_kernel_wasserstein_minfunc
% Inputs: K is kernel matrix, x_idx is logical index for one samples
% Purpose: compute the max-sliced kernel Wasserstein-2 metric
% \omega(*) = \sum_i \alpha_i K(z_i,*) 
% Outputs: div  (objective value), alpha (parameters of the solution)

if nargin<3
    L = size(K,1);
    K_L = K +1e-9*eye(L);
else
    K_L = K(z_idx,z_idx);
    L = size(K_L,1);
    K_L = K_L + 1e-9*eye(L);
end

nMonte = 20; % number of random initializations
if nargin==4
    if isfield(params,'nMonte') 
        nMonte = params.nMonte;
    end
end

m = sum(x_idx);
n = size(K,1) - m;
K = (K + K.')/2;
K_X_Z  = K(  x_idx,  z_idx);
K_Y_Z  = K( ~x_idx, z_idx);

A = K_X_Z;
B = K_Y_Z;
R = 1/m*(K_X_Z'*K_X_Z) + 1/n*(K_Y_Z'*K_Y_Z);

numer = @(alpha) alpha'*R*alpha...
    -2*(alpha'*K_X_Z')*sortOT(K_X_Z*alpha, K_Y_Z*alpha)*(K_Y_Z*alpha);
%h2 = @(alpha) trace(sortOT(K_X_Z*alpha, K_Y_Z*alpha)'*pdist2(K_X_Z*alpha, K_Y_Z*alpha,'squaredeuclidean'));
% abs(h-h2) %check 

denom = @(alpha) alpha'*K_L*alpha; 
obj = @(x) numer(x)/denom(x); %to Maximize

objs = nan(nMonte,1); % to collect objective across different
% initial starting points
for monte_ii = 1:nMonte
    % INITIALIZATION FOR PRIMAL
    if monte_ii == 1 %barycenter of transportation polytope (max entropy)
            Q = mean(K_X_Z,1)'*mean(K_Y_Z,1);
            Q = R - Q - Q';
            [alpha_k,~] =  eigs(Q+Q', K_L, 1);
    else
        alpha_k = randn(N,1); % random
    end
    % ensure that it is feasible
    alpha_k = alpha_k / sqrt(alpha_k'*K_L*alpha_k);
    options=[];
    options.Display='off';
    % END INITIALIZATION
    
    % MAIN LOOP
    costfungrad = @(x) do_unconstrained_msw2(x,K_L,R,A,B);
    [x,~] = minFunc(costfungrad,alpha_k,options);
    x = x / sqrt(x'*K_L*x);
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
fprintf('Obj= %.7f \n\n', div)
end
   

function [f,grad] = do_unconstrained_msw2(x,K,R,A,B)
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
    
