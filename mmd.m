function [V,divs,alphas,D1] = mmd(K,x_idx,tests)

K = (K + K.')/2; %ensure symmetric  (doesn't ensure PSD)

m = sum(x_idx); % sample size for X
n = size(K,1) - m; % sample size for Y

alphas = 1/m*x_idx - 1/n*(~x_idx);
div_mmd = alphas'*K*alphas;
%div_mmd_max_sliced = sqrt(max(0,sum((mean(K_X_Z,1) - mean(K_Y_Z,1)).^2)));  %finding mean diff
divs = div_mmd;% div_mmd_max_sliced];

V = K*alphas;


if  nargin> 2 && tests > 1
    p1=zeros(tests,1);
    for t=1:tests
        new_idx = x_idx(randperm(n+m));
        [~,p1(t),~,~] = mmd(K,new_idx);
    end
    if nargout <= 4
        %D1 = mean(p1<divs); % proportion of the shuffle less than D_22
        %D2 = 1 - mean(D2>p2);
        D1 = p1;
    end
else
    D1 = 0;
end


end