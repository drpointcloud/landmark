m = 700;
n = 700;
n_clusters = 10;

prevalance_of_missing = 0.025;

[x,y] = make_clusters(m,n,n_clusters,prevalance_of_missing);

figure(1),clf
plot(x(:,1),x(:,2),'o'), hold all
plot(y(:,1),y(:,2),'x')

z = [x;y];
x_idx = 1==[ones(m,1);zeros(n,1)];
K = make_gaussian_kernel(z);


figure(2),clf
imagesc(K)


[idx,div] = MLW(K,x_idx,1);
disp(div)
figure(1)
plot(z(idx,1),z(idx,2),'k*','markersize',40,'linewidth',2)
%%
ns = round(logspace(2,log10(5000),15));
n_clusters = 8;

times = nan(numel(ns),4);
divs = zeros(numel(ns),4);

for i = 1:numel(ns)
    n = ns(i);
    m = n;
    [x,y] = make_clusters(m,n,n_clusters,0);
    z = [x;y];
    x_idx = 1==[ones(m,1);zeros(n,1)];

    if n<600
        start = tic;
        D2 = make_sqeuclidean(z);
        div0 = discrete_W2(D2,x_idx,true);
        times(i,1) = toc(start);
    end    
%     start = tic;
%     D2 = make_sqeuclidean(z);
%     div0 = discrete_W2(D2,x_idx,false);
%     times(i,2) =  toc(start);

    start = tic;
    K = make_gaussian_kernel(z);
    div1 = MMD(K,x_idx);
    times(i,3) =  toc(start);

    start = tic;
    K = make_gaussian_kernel(z);
    [~,div2] = MLW(K,x_idx);
    times(i,4) =  toc(start);
    disp([i,times(i,:)])
end
%%

figure()
plot(ns,times(:,1),':*','linewidth',2)
hold all
plot(ns,times(:,4),':^','linewidth',2)
plot(ns,times(:,3),':x','linewidth',2)
ylabel('Time (s)')
xlabel('Sample size $m=n$','interpreter','latex')
legend('Discrete W2 (w/ Borlin''s Hungarian)','MLW','MMD')
title('MATLAB')
%%
n_permutations = 1000;

tic
[idx2,div2,pvalue2] = MLW(K,x_idx,1,n_permutations);
t = toc;
disp([idx2,div2,pvalue2,t])


%%
function K = make_gaussian_kernel(z)
    D = squareform(pdist(z));
    sigma = median(D+triu(0./zeros(size(D))),'all','omitnan'); % uses median kernel size 
    K = exp(-0.5/sigma^2*D.^2);
end

function D2 = make_sqeuclidean(z)
    D2 = squareform(pdist(z)).^2;
end

function [div,pvalue] = MMD(K,x_idx,permutations)
    m = sum(x_idx);
    n = size(K,1)-m;
    alphas = 1/m*x_idx - 1/n*(~x_idx);
    div = sqrt(max(0,alphas'*K*alphas));
    if nargin>2 && permutations>0 && nargout>1
        d=zeros(permutations,1);
        for b=1:permutations
            new_x_idx = x_idx(randperm(n+m));
            alphas = 1/m*new_x_idx - 1/n*(~new_x_idx);
            div(b) = sqrt(max(0,alphas'*K*alphas));
        end
        pvalue = mean(d>=div);
    end
end
    


function W2 = discrete_W2(D2,x_idx,use_hungarian)
%C = sum(X.^2,2) + sum(Y.^2,2).' -2*X*Y'; % squared Euclidean
    m = sum(x_idx);
    n = size(D2,1)-m;
    mu = 1/m*ones(m,1);
    nu = 1/n*ones(n,1);
    C = D2(x_idx,~x_idx);
    
    if m==n && nargin<2 || (m==n) && use_hungarian
        [~,obj_val]=hungarian(C);
        W2 = sqrt(obj_val);
    else
        % Peyre's MATLAB https://nbviewer.jupyter.org/github/gpeyre/numerical-tours/blob/master/matlab/optimaltransp_1_linprog.ipynb
        flat = @(x)x(:);
        Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
                     flat(reshape(1:n0*n1,n0,n1) ), ...
                     ones(n0*n1,1) );
        Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
                     flat(reshape(1:n0*n1,n0,n1)' ), ...
                     ones(n0*n1,1) );
        Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];

        maxit = 1e4; tol = 1e-9;
        otransp = @(C,p0,p1)reshape( perform_linprog( ...
                Sigma(length(p0),length(p1)), ...
                [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );

        W2 = sqrt(flat(C)'*flat(otransp(C,mu,nu)));
    end
end

function [x,y] = make_clusters(m,n,K,prevalance_of_missing)
    if nargin<4
    prevalance_of_missing = 0.5/K;
    end

    angles = linspace(0,2*pi,K+1);
    angles = angles(1:K);
    clusters = 8*[cos(angles);sin(angles)]';


    x_i = random('unid',K,m,1);
    x = randn(m,2) + clusters(x_i,:);
    w = (1-prevalance_of_missing)/(K-1)*ones(1,K-1); w= [w,1-sum(w)];

    y_i = randsample(numel(angles),n,true,w);
    y = randn(n,2) + clusters(y_i,:);
end