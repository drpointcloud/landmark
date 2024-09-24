m = 400;
n = 400;
n_clusters = 8;

prevalance_of_missing = 0.5/n_clusters;

[x,y] = make_clusters(m,n,n_clusters,prevalance_of_missing);

figure(1),clf
plot(x(:,1),x(:,2),'o'), hold all
plot(y(:,1),y(:,2),'x')

z = [x;y];
x_idx = 1==[ones(m,1);zeros(n,1)];
K = make_gaussian_kernel(z);


figure(21),clf
imagesc(K)


% [~,div1,alphas,~] = L_MSKW(K,x_idx);
% idx1 = find(alphas);
% disp([idx1,div1])

[idx2,div2] = MLW(K,x_idx);
disp([idx2,div2])

n_permutations = 1000;
% tic 
% [~,div1,alphas,divs] = L_MSKW(K,x_idx,n_permutations);
% idx1 = find(alphas);
% pvalue1 = mean(divs>div1);
% t = toc;
% disp([idx1,div1,pvalue1,t])

figure(11)
plot(z(idx1,1),z(idx1,2),'*','markersize',40)


tic
[idx2,div2,pvalue2] = MLW(K,x_idx,2,n_permutations);
t = toc;
disp([idx2,div2,pvalue2,t])



%%
ns = round(logspace(2, 3.5,5));
n_permutations = 2000;
times = zeros(numel(ns),2);
for n_i = 1:numel(ns)
    n = ns(n_i);
    [x,y] = make_clusters(n,n,n_clusters,prevalance_of_missing);
    z = [x;y];
    x_idx = 1==[ones(n,1);zeros(n,1)];
    K = make_gaussian_kernel(z);
    
    tic
    [idx2,div2,pvalue2] = MLW(K,x_idx,2,n_permutations,0);
    times(n_i,1) = toc;
    disp([idx2,div2,pvalue2,times(n_i,1)])
    tic
    [idx2,div2,pvalue2] = MLW(K,x_idx,2,n_permutations,1);
    times(n_i,2) = toc;
    disp([idx2,div2,pvalue2,times(n_i,2)])
end
figure(31),clf
plot(ns,times)

%%
function K = make_gaussian_kernel(z)
    D = squareform(pdist(z));
    sigma = median(D+triu(0./zeros(size(D))),'all','omitnan');
    K = exp(-0.5/sigma^2*D.^2);
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