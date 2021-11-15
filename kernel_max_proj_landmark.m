function [V,div,alphas,D1] = kernel_max_proj_landmark(K,x_idx,method,tests)
% kernel_max_proj finds projection that maximizes a divergence
% Inputs: K is a kernel matrx, x_idx is a logical index of one sample
%         (n by n)            (n-length)
%  such that K(x_idx,x_idx) = K_X
%       method is a string
% Purpose: compute and evaluate kernel projections
% Outputs: V projection coordinates, divs  (objective value),
%          alphas (parameters of the  projects)
%  V = K*alphas
% Austin J. Brockmeier (ajbrockmeier@gmail.com)
% 9/22/2020
if iscell(method)
    div_measure = method{1};
    %opt_method = method{2};
else
    div_measure = method;
    %opt_method = 'fastest';
end


%m = sum(x_idx); % sample size for X
%n = size(K,1) - m; % sample size for Y
%
K = (K + K.')/2; % ensure symmetric
%K_X_Z  = K(  x_idx,  :); % Z = X union Y
%K_Y_Z  = K( ~x_idx, :);

%A = sqrt(1/n)*K_Y_Z; % D_\nu^\frac{1}{2}  K_{YZ}
%B = sqrt(1/m)*K_X_Z; % D_\mu^\frac{1}{2}  K_{XZ}


%%
switch lower(div_measure)
        
    case {'l-w2','w2','l-w2-max','l-w2-mean','max-w2'}% Max-sliced kernel Wasserstein 2
        [V,divs,alphas,D1] = L_MSKW(K,x_idx,tests);
        switch lower(div_measure)
            case {'l-w2-max'}  % D(:,1) for landmark max-div_values
                div = divs(1); 
                D1 = D1(:,1);
            case {'l-w2-mean'} % D(:,2) for landmark mean-div_values
                div = divs(2);
                D1 = D1(:,2);
            otherwise
                div = divs(1);
                D1 = D1(:,1);
        end
        %%
        
    case {'l-bures-max','l-bures','l-bures-mean','bures','max-bures','max-b'}
        [V,divs,alphas,D1] = L_MSKB_one_side(K,x_idx,tests);
        switch lower(div_measure)
            case {'l-bures-max'} % D(:,1) for landmark max-div_values
                div = divs(1);
                D1 = D1(:,1);
            case {'l-bures-mean'} % D(:,2) for landmark mean-div_values
                div = divs(2);
                D1 = D1(:,2);
            otherwise
                div = divs(1);
                D1 = D1(:,1);
        end
        %%
    otherwise
        error('unknown method  %s', div_measure)
end

%divs = max(divs);
%[divs,idx]= max(divs,[],'omitnan');
%V = K*alphas;
%V = V(:,idx);

%%
% if  nargin> 4 && tests > 1
%     p1=zeros(tests,2);
%     for t=1:tests
%         new_idx = x_idx(randperm(n+m));
%         [~,p1(t,:),~,~] = kernel_max_proj(K,new_idx,method);
%     end
%     if nargout <= 5
%         %D1 = mean(p1<divs); % proportion of the shuffle less than D_22
%         %D2 = 1 - mean(D2>p2);
%         D1 = p1;
%     end
% else
%     D1 = 0;
% end



end









%%















%%


