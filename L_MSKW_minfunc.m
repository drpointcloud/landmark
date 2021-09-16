function [V,divs,alphas,D1] = L_MSKW_minfunc(K,x_idx,tests)
% max_sliced_kernel_wasserstein_minfunc
% Inputs: K is kernel matrix, x_idx is logical index for one samples
% Purpose: compute the max-sliced kernel Wasserstein-2 metric
% \omega(*) = \sum_i \alpha_i K(z_i,*) 
% Outputs: div  (objective value), alpha (parameters of the solution)
% Austin J. Brockmeier (ajbrockmeier@gmail.com)
% 8/11/2020

%N = size(K,1);
% nMonte = 20; % number of random initializations
% if nargin==3
%     if isfield(params,'nMonte') 
%         nMonte = params.nMonte;
%     end
% end

% D_w = ||\omega||_2^2 

m = sum(x_idx);
n = size(K,1) - m;
Niter = m+n;

K = (K + K.')/2; % ensure symmetric
K_X_Z  = K(  x_idx, :);
K_Y_Z  = K( ~x_idx, :);
% A = K_X_Z;
% B = K_Y_Z;

% Kxz = @(alpha)(A*alpha);
% Kyz = @(alpha)(B*alpha);
%R = 1/m*(K_X_Z'*K_X_Z) + 1/n*(K_Y_Z'*K_Y_Z);

% h = @(z) 1/m*(Kxz(z)' * Kxz(z)) + 1/n*(Kyz(z)' * Kyz(z))...
%          -2*(Kxz(z)') * sortOT(Kxz(z), Kyz(z)) * (Kyz(z));
% obj = @(x) sqrt(h(x));

R = 1/m*(K_X_Z'*K_X_Z) + 1/n*(K_Y_Z'*K_Y_Z);

h = @(alpha) alpha'*R*alpha...
    -2*(alpha'*K_X_Z')*sortOT(K_X_Z*alpha, K_Y_Z*alpha)*(K_Y_Z*alpha);
%h2 = @(alpha) trace(sortOT(K_X_Z*alpha, K_Y_Z*alpha)'*pdist2(K_X_Z*alpha, K_Y_Z*alpha,'squaredeuclidean'));
% h==h2

g = @(alpha) alpha'*K*alpha; 
obj = @(x) h(x)/g(x);
%%

nus = nan(Niter,1);
for niter_ii=1:Niter % Landmark based iteration
    alpha_k = zeros(Niter,1); % added
    alpha_k(niter_ii) = 1; % added
    nus(niter_ii) = obj(alpha_k);
    %%
end
%%
alphas = zeros(Niter,1);

[div , idx] = max(nus,[],'omitnan');
alphas(idx) = 1;
divs = sqrt(div);
V = K*alphas;

if  nargin >2 && tests >1
    p1=zeros(tests,1);
    parfor t=1:tests
        new_idx = x_idx(randperm(n+m));
        [~,p1(t),~,~] = L_MSKW_minfunc(K,new_idx);
    end
    if nargout <= 4
        %D1 = mean(p1<divs); % proportion of the shuffle less than D_22
        %D2 = 1 - mean(D2>p2);
        D1 = p1;
    end
else
    D1 =0;
end




   




   

