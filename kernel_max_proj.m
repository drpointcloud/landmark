function [V,divs,alphas,D1] = kernel_max_proj(K,x_idx,method,tests)
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
    opt_method = method{2};
else
    div_measure = method;
    opt_method = 'fastest'; 
end


m = sum(x_idx); % sample size for X
n = size(K,1) - m; % sample size for Y

% K_X_Z  = A;
% K_Y_Z  = B;

K = (K + K.')/2; %ensure symmetric  (doesn't ensure PSD)
    K_X_Z  = K(  x_idx,  :);
    K_Y_Z  = K( ~x_idx, :);


A = sqrt(1/n)*K_Y_Z; % D_\nu^\frac{1}{2}  K_{YZ}
B = sqrt(1/m)*K_X_Z; % D_\mu^\frac{1}{2}  K_{XZ}

switch lower(div_measure)
    case 'mmd' % Maximum mean discrepancy 
        alphas = 1/m*x_idx - 1/n*(~x_idx);
        div_mmd = alphas'*K*alphas;
        %div_mmd_max_sliced = sqrt(max(0,sum((mean(K_X_Z,1) - mean(K_Y_Z,1)).^2)));  %finding mean diff
        divs = div_mmd;% div_mmd_max_sliced];
    case {'w2','max-w2'}% Max-sliced kernel Wasserstein 2
        [div,alpha] = max_sliced_kernel_wasserstein_minfunc(K,x_idx);
        divs = div;
        alphas = alpha;
    case {'tv','kolmogorov','max-tv','max-kolmogorov'}
        % max-sliced kernel TV
        AA = (A'*A);
        BB = (B'*B);
        [V,D]=eigs(AA-BB,K + 1e-9*eye(n+m), 1,'bothendsreal');
        alphas = V*sign(D);
        divs = 0.5*diag(abs(D));
    case {'gw','max-gw','max-gauss-wasserstein','gauss-wasserstein'}
        % max-sliced Gauss Wasserstein (Lower bound)
        [divs,alphas] = max_gw_minfunc(K,x_idx);
    case {'kfda'} %Kernel Fisher discriminant analysis
        lambda = 1e-6;
        b = (mean(K_Y_Z,1) - mean(K_X_Z,1))';
        A = bsxfun(@minus,K_Y_Z,mean(K_Y_Z,1));
        B = bsxfun(@minus,K_X_Z,mean(K_X_Z,1));
        N = A'*A + B'*B;
        alpha = (N + lambda*eye(n+m))\b;
        divs = (b'*alpha)^2/(alpha'*N*alpha);
        alphas = alpha;
    case {'bures','max-bures','max-b','max-gw-u','gw-u','gw-upper'}
        % Max-sliced Bures or max-sliced Gauss Wasserstein (Upper bound)
        switch lower(div_measure)
            case {'max-gw-u','gw-u','gw-upper'}
                %A = sqrt(1/n)*bsxfun(@minus,K_Y_Z,mean(K_Y_Z,1));
                %B = sqrt(1/m)*bsxfun(@minus,K_X_Z,mean(K_X_Z,1));
        end
        switch lower(opt_method)
            case {'fastest','minfunc'}
                one_side_max_bures_opt = @one_side_max_bures_minfunc;
            case 'cvx'
                one_side_max_bures_opt = @one_side_max_bures_cvx;
        end
        try
            [div1,alpha1] = one_side_max_bures_opt(K,x_idx,tests);
            [div2,alpha2] = one_side_max_bures_opt(K,~x_idx,tests);
        catch
            [div1,alpha1] = one_side_max_bures_opt(K + 1e-9*eye(n+m),x_idx,tests);
            [div2,alpha2] = one_side_max_bures_opt(K + 1e-9*eye(n+m),~x_idx,tests);
        end
%         alphas = [alpha1,alpha2];
%         divs = [div1,div2];
        if div1>div2
            alphas = alpha1;
        else
            alphas =alpha2;
        end
        divs = max(div1,div2);
        
    otherwise
        error('unknown method  %s', div_measure)
end

switch lower(div_measure)
    case {'max-gw-u','gw-u','gw-upper'}
        % Max-sliced Gauss Wasserstein (Upper bound)
        % Find difference in mean 
        diff = (mean(K_X_Z,1) - mean(K_Y_Z,1))';
        alpha_mmd = cat(1,1/m*ones(size(K_X_Z,1),1),-1/n*ones(size(K_Y_Z,1),1));
        div_mmd = diff'*alpha_mmd/sqrt(alpha_mmd'*K*alpha_mmd);
        [div2,idx]=max(divs); % get the Bures distance 
        alpha2 = alphas(:,idx);
        alphas = [alpha_mmd,alpha2];
        divs = [div_mmd,div2];
end

divs = max(divs);
%[divs,idx]= max(divs,[],'omitnan');
V = K*alphas;
%V = V(:,idx);
%%
if  nargin> 3 && tests > 1
    p1=zeros(tests,1);
    for t=1:tests
        new_idx = x_idx(randperm(n+m));
        [~,p1(t),~,~] = kernel_max_proj(K,new_idx,method);
    end
    if nargout <= 4
        %D1 = mean(p1<divs); % proportion of the shuffle less than D_22
        %D2 = 1 - mean(D2>p2);
        D1 = p1;
    end
else
    D1 = 0;
end

% varargout{1} = V;
% varargout{2} = divs;
% varargout{3} = alphas;
% varargout{4} = D1;









