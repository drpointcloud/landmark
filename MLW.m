function [landmark_idx, div,pvalue] = MLW(K,x_idx,p_W,permutations,do_sort_only_once)
% Inputs: K is kernel matrix, x_idx is logical index for first (X) sample
%           p_W is the expeonent, and tests is how many random permutations
% Purpose: compute the max-sliced kernel landmark Wasserstein-p metric
% Outputs: landmark_idx, div  (objective value), p-value
% Austin J. Brockmeier (ajbrock@udel.edu)
% 8/05/2024

if nargin<3
    p_W = 2;
end

m = sum(x_idx); % sample size for X
n = size(K,1) - m; % sample size for Y

K = (K + K.')/2; %ensure symmetric  (doesn't ensure PSD)

K_X_Z  = K(  x_idx,  :);
K_Y_Z  = K( ~x_idx, :);


[landmark_idx,div] = MLW_sorted(sort(K_X_Z),sort(K_Y_Z),p_W);

if  nargin>3 && permutations >1
    if nargin == 4
        do_sort_only_once = 0;
    end
    d=zeros(permutations,1);
    if do_sort_only_once
        [K_acute, idx_per_landmark] = sort(K);
    end
    for b=1:permutations
        new_x_idx = x_idx(randperm(n+m));
        if do_sort_only_once
            K_acute_X = reshape(K_acute(new_x_idx(idx_per_landmark)), size(K_X_Z));
            K_acute_Y = reshape(K_acute(~new_x_idx(idx_per_landmark)), size(K_Y_Z));
            [~,d(b)] = MLW_sorted(K_acute_X, K_acute_Y, p_W);
        else
            [~,d(b)] = MLW(K, new_x_idx, p_W);
        end
    end
    pvalue = mean(d>div);
else
    pvalue = nan;
end

function [idx,div] = MLW_sorted(K_X_Z,K_Y_Z,p_W)
    m = size(K_X_Z,1);
    n = size(K_Y_Z,1);
    
    if m==n  % assumes the X and Y are equal size
        landmark_divs =  mean( abs(K_X_Z - K_Y_Z).^p_W , 1);
    else %if m is not equal to n, mass spliting applied for exact corresponding
        P = sortOT((1:m)',(1:n)'); % P is sparse
        [i,j,p] = find(P);
        R_X = sparse(1:numel(p),i,1,numel(p),m);
        R_Y = sparse(1:numel(p),j,1,numel(p),n);
        landmark_divs = p(:)'*abs(R_X*K_X_Z-R_Y*K_Y_Z).^p_W;
    end
    [div_max,idx] = max(landmark_divs);
    div = div_max^(1/p_W);



   

