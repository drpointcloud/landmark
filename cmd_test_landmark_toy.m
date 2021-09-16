
clc
clear
%%

%% Create toy data 1
% pz = [10 20 30 40 50]/2;
% [px,py]=meshgrid(pz,pz);
% centers = [px(:),py(:)];
% X_P = @(m) randn(m,2) + centers(randi(size(centers,1),m,1),:);
% C_Q = @(ep) [1, (ep - 1)/(ep + 1); (ep - 1)/(ep + 1), 1];
% Y_P = @(m,ep) mvnrnd([0;0],C_Q(ep),m)+ 1.5*centers(randi(size(centers,1),m,1),:)...
%     +2*(max(max(X_P(m)))- mean(mean(X_P(m))));
% X_P = @(m) mvnrnd([0;0],C_Q(1),m)+centers(randi(size(centers,1),m,1),:);

%% Create toy data 2
X_P = @(m) randn(m,2);
C_Q = @(ep) [1, (ep - 1)/(ep + 1); (ep - 1)/(ep + 1), 1];
Y_P = @(m,ep) mvnrnd([0;0],C_Q(ep),m);%(max(max(X_P(m)))- mean(mean(X_P(m))));
%centers = @(d) cat(1,[0.5,zeros(1,d-1)],[0.0,0.5,zeros(1,d-2)]);
%Y_P = @(m,d) randn(m,d) + sparse(1:m,randi(2,1,m),1,m,2)*centers(d)+4 ;
%%


m = 50;  % total points
N = 2*m;
nperms = 100;
nMonte = 100;
eps = linspace(1,25,10);%log(3.^(2:2:40));

%%
method1 = {
    { @(K,x_idx) L_MSKB_one_side_minfunc(K,x_idx,nperms), 'L-B'};
    { @(K,x_idx) L_MSKW_minfunc(K,x_idx,nperms), 'L-W2'};
    }; 
method2 = {
    %{ @(K,x_idx,method) kernel_max_proj(K,x_idx,method,nperms), 'W2'};
    %{ @(K,x_idx,method) kernel_max_proj(K,x_idx,method,nperms), 'Bures'};
    { @(K,x_idx,method) kernel_max_proj(K,x_idx,method,nperms), 'MMD'};
%     { @(K,x_idx,method) kernel_max_proj(K,x_idx,method,nperms), 'Kolmogorov'};
%     {'Bures','Max-Sliced Kernel Bures',@(x) abs(x)}; % requires minFunc
%     {'W2','Max-Sliced Kernel W2',@(x) x}; % requires minFunc
%     {'Kolmogorov','Max-Sliced Kernel TV',@(x) abs(x)};
%     {'MMD','MMD',@(x) x};
};
total_methods = cat(1,method1,method2);
if isunix
    slash = '/';
else
    slash = '\';
end
% methods = {'L-Bures','L-W2','Bures','W2','Kolmogorov','MMD'};
%%
landmark = zeros(numel(eps),nMonte,numel(total_methods));
obj = landmark;
d = zeros(nperms,numel(eps),nMonte,numel(total_methods));
witness = zeros([N size(obj)]);
betahat = landmark;
%% print the progress
aaa=0;
timestart = tic;
perc = numel(eps)*nMonte*numel(total_methods);
progress = '..........'; 

%%
for eps_ii = 1:numel(eps)
    epsilon = eps(eps_ii);
    
    for nMonte_ii = 1:nMonte
        tic

        X = X_P(m);
        Y = Y_P(m,epsilon);

        Z = double(cat(1,X,Y)); % combine two samples
        x_idx = cat(1,ones(size(X,1),1),zeros(size(Y,1),1))==1; %bool indicator
        [K,~] = gaussian_kernel(Z,'median');
        K = (K + K.')/2; % ensure symmetric

        %% method 1
        offset =0;
        for method_ii = 1:numel(method1)
            aaa=aaa+1;
            method = method1{method_ii}{1};
            [V,div,alphas,D1] =  method(K,x_idx);
            idx = find(alphas);
            landmark(eps_ii,nMonte_ii,method_ii+offset) = idx;
            obj(eps_ii,nMonte_ii,method_ii+offset) = div;
            witness(:,eps_ii,nMonte_ii,method_ii+offset) = V;
            d(:,eps_ii,nMonte_ii,method_ii+offset) = D1;
            %cv = quantile(D1,1-alpha);
            betahat(eps_ii,nMonte_ii,method_ii+offset) = mean(D1>=div);
        end
        
        %% method 2
        offset = numel(method1);
        for method_ii = 1:numel(method2)
            aaa=aaa+1;
            method_way = method2{method_ii}{1};
            method = method2{method_ii}{2};
            
            [V,div,alphas,D1] = method_way(K,x_idx,method);
            obj(eps_ii,nMonte_ii,method_ii+offset) = div;
            witness(:,eps_ii,nMonte_ii,method_ii+offset) = V;
            d(:,eps_ii,nMonte_ii,method_ii+offset) = D1;
            %cv = quantile(D1,1-alpha);
            betahat(eps_ii,nMonte_ii,method_ii+offset) = mean(D1>=div);            
        end
        clc
        progress = progress_print(aaa,perc,timestart,progress);
        %toc
        %kk=kk+1;
        %fprintf('%f (%d)\n', toc,kk)
        
    end
end

%% plot divergences
linestyle= {'-','-',':',':',':','--','-',':','--','-'};
%%
% nan(numel(total_methods),numel(classes),nmonte);
alpha = [0.05 0.1 0.15];
figure(2);clf
for i = 1:numel(alpha)
    alpha_ii = alpha(i);
    % cv0 = squeeze(quantile(obj,alpha,2));
    % cv1 = squeeze(quantile(obj,1-alpha));
    powhat = squeeze((mean(betahat<alpha_ii,2)));
    % DD = dq;
    % D0 = squeeze(mean(DD,2));
    % D1 = arrayfun(@(i)reshape(D0(:,i:3:end),Nmthds,[]),(1:3),'uni',0);
    % D2 = permute(cell2mat(permute(D1,[1,3,2])),[1,3,2]);    
    subplot(1,3,i)
    semilogy(powhat,'linewidth',3);
    set(gca,'fontsize',10)
    set(gca,'xgrid','on')
    title(['Rejection rate at \alpha =',sprintf('%0.2f\n',alpha_ii)])
    xlabel('\epsilon','FontSize',15,'FontWeight','Bold')
    xticks(1:numel(eps))
    set(gca,'XTickLabel',sprintf('%0.0f\n',eps))
    ylim([0 1.1])
    %title(['power test of toy data \alpha =',num2str(alpha_ii)])
    legend('L-Bures','L-W2','MMD','Location','southeast')%,'Unbiased MMD')
end
set(gcf, 'Color', 'w');
export_fig power_toy.pdf
%% plot divergences %as a function of iteration
%as a function of iteration
figure(10);clf
subplot(121)

divergence = obj(1,:,:);
meanA =  squeeze(mean(divergence,2));
meanB = mean(meanA);
kk=0;
for method_ii =  1:numel(total_methods)
    kk= kk +1;
    hold on
    plot(1:nMonte,divergence(1,:,method_ii)+(meanB-meanA(method_ii)),linestyle{kk},'linewidth',2)

end
% legend(cellfun(@(x) x{1},total_methods,'uni',0),'location','southeast')
xlabel('Samples are drawn for 250 times')
ylabel('Normalized divergence')
title('Divergence variations for 250 draws')

%as a function of epsilon
subplot(122)
divergence = obj;
divs =  squeeze(mean(divergence,2));
meanA = mean(divs,1);
meanB = mean(meanA);
kk=0;
for method_ii =  1:numel(total_methods)
    kk= kk +1;
    plot(divs(:,method_ii)+(meanB-meanA(method_ii)),linestyle{kk},'linewidth',2)
    hold on
end
hold off
legend(cellfun(@(x) x{2},total_methods,'uni',0),'location','southeast')
xlabel('\epsilon','fontsize',13,'FontWeight','bold')
ylabel('Normalized divergence')
title('Divergence variations for \epsilon\in[2,40]')


%% Box plots
%methods = {'L-Bures','L-W2','MMD'};
divergence = obj;
divs =  squeeze(mean(divergence,2));
meanA =  squeeze(mean(divergence(1,:,:),2));

bxpltnMont = zeros(size(divergence,2),numel(total_methods));
bxpltEpsil = zeros(size(divergence,1),numel(total_methods));

methods_name = cell(1,numel(total_methods));
for method_ii=1:numel(total_methods)
    
    bxpltnMont(:,method_ii) = divergence(1,:,method_ii)+(1-meanA(method_ii));
    bxpltEpsil(:,method_ii) = divs(:,        method_ii);%+(mean(meanB)-meanB(method_ii));
    methods_name{method_ii} = total_methods{method_ii}{2};
end

figure(11);clf

subplot(121)
boxplot(bxpltnMont,{methods_name})
xlabel('(X\sim\mu,Y\sim\nu) are drawn for 250 times')
ylabel('Divergence variation')
% title('Divergence variations for 250 draws')

%as a function of epsilon
subplot(122)
boxplot(bxpltEpsil,{methods_name})
xlabel(['Divergence variations for \epsilon\in[',num2str(min(eps)),',',num2str(max(eps)),']'])
ylabel('Divergence value')

%% Plot input data
figure(30),clf
neps = numel(eps);
ap = ceil(sqrt(2*neps)); %to show all plots together as # epsilon varies
aq = ceil(2*neps/ap);
for eps_ii = 1:numel(eps)
    epsilon = eps(eps_ii);
    X = X_P(m);
    Y = Y_P(m,epsilon);
    
    subplot(ap,aq,eps_ii)
    h=plot(X(:,1),X(:,2),'.','linewidth',2,'MarkerSize',10);
    set(h,'color',[1 .7 .2])
    set(gca,'xgrid','on','ygrid','on')
    axis square
    xlabel('X_1');ylabel('X_2')
    
    
    subplot(ap,aq,eps_ii+neps)
    h=plot(Y(:,1),Y(:,2),'.','linewidth',2,'MarkerSize',10);
    set(h,'color',[.4 1 1])
    set(gca,'xgrid','on','ygrid','on')
    axis square
    xlabel('Y_1');ylabel('Y_2')
    title(['\epsilon=',num2str(round(epsilon))])
end

%% %% plot the vertcal lines - Landmark Bures/Wasserstein
epsilon = 1;
X = X_P(m);
Y = Y_P(m,epsilon);
Z = cat(1,X,Y);
rp = [landmark(1,:,1)' landmark(1,:,2)'];
nM = size(rp,1);

figure(32);clf

p = 10; % it will be pxp size tile.
q = p-1; % the big image is qxq
t = tiledlayout(p,p); % Requires R2019b or later

ax3 = nexttile(1,[q q]);
plot(ax3,X(:,1),X(:,2),'.','linewidth',3,'MarkerSize',20); 
hold on
plot(ax3,Y(:,1),Y(:,2),'.','linewidth',3,'MarkerSize',20);
hold('on');
line([8 8],[-3 -3],'Color','red','LineStyle','-');
line([8 8],[-3 -3],'Color','black','LineStyle','-');
legend({'X','Y','L-Bures',' L-W2'}, 'Location', 'southeast');

ax4 = nexttile(p,[q 1]);
plot(ax4,cat(1,zeros(1,nM),ones(1,nM)),cat(1,Z(rp(:,2),1).',Z(rp(:,2),1).'),'-r','linewidth',1);
hold on
plot(ax4,cat(1,ones(1,nM),2*ones(1,nM)),cat(1,Z(rp(:,1),1).',Z(rp(:,1),1).'),'-k','linewidth',1);


ax5 = nexttile(p*(p-1)+1,[1 q]);
plot(ax5,cat(1,Z(rp(:,2),1).',Z(rp(:,2),1).'),cat(1,zeros(1,nM),ones(1,nM)),'-r','linewidth',1);
hold on
plot(ax5,cat(1,Z(rp(:,2),1).',Z(rp(:,2),1).'),cat(1,ones(1,nM),2*ones(1,nM)),'-k','linewidth',1);

linkaxes([ax3,ax4],'y');
linkaxes([ax3,ax5],'x');
xticklabels(ax3,{})
yticklabels(ax3,{})
title(t,'Corresponding location of landmarks for 250 trials')
t.TileSpacing = 'compact';
% export_fig test3.png



%% plot vertical lines for each epsilon
figure(33),clf
Nep = numel(eps)+1;
ax10 = zeros(Nep,1);
q = 5;% main image is qxq
p = numel(eps) +q; %tile is pxp size
t = tiledlayout(p,q); % Requires R2019b or later

Y(:,1) = Y(:,1) + max(X(:,1))-min(Y(:,1));
X1 = X(:,1); Y1 = Y(:,1);
Z = cat(1,X1,Y1);

ax10(1) = nexttile(1,[q q]);
plot(X(:,1),X(:,2),'.','Color',[0 .5 .7],'linewidth',2,'MarkerSize',15); hold on;
hold on
plot(Y(:,1),Y(:,2),'.','Color',[.8 .3 .01],'linewidth',2,'MarkerSize',15); hold on;
hold('on');

line([8 8],[-3 -3],'Color','red','LineStyle','-');
line([8 8],[-3 -3],'Color','black','LineStyle','-');
lgd = legend({'X','Y','L-Bures',' L-W2'}, 'Location', 'southeast','orientation','horizontal');
lgd.NumColumns = 2;
legend('boxoff')
hold off
rp = landmark(:,:,1); %landmark bures
np = landmark(:,:,2); % landmark W2
nM = size(rp,2);

for i = 2:Nep
    ax10(i) = nexttile((i-1)*q+q*(q-1)+1,[1 q]);
    plot(ax10(i),cat(1,Z(rp(i-1,:)).',Z(rp(i-1,:)).'),...
        cat(1,zeros(1,nM),ones(1,nM)),'-r','linewidth',1);hold on
    plot(ax10(i),cat(1,Z(np(i-1,:)).',Z(np(i-1,:)).'),...
        cat(1,ones(1,nM),2*ones(1,nM)),'-k','linewidth',1);
    
    xlim([min(min(Z(:,1))) max(max(Z(:,1)))])
    %linkaxes([ax10(i-1),ax10(i)],'x');
    xticklabels(ax10(i-1),{})
    yticklabels(ax10(i),{})
    lgd = legend(['$\epsilon$=',sprintf('%.2f',eps(i-1))],...
        'Location', 'west','Interpreter','latex','FontSize',12);
    lgd.FontWeight = 'bold';
    %legend('boxoff')
end
title(t,'Corresponding location of landmarks')
t.TileSpacing = 'none';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Print the progress
function string = progress_print(aaa,perc,timestart,string)
    tEnd = toc(timestart); 
    percentage = aaa/perc*100;
    remt = 100*tEnd/percentage -tEnd;
    if fix(percentage/10) ~= 0
        string(fix(percentage/10)) = '*';
    end
    fprintf('\n[%s]', string);
    fprintf(' %0.1f%% is completed...\n',percentage)
    disp(['elapsed time:',datestr(datenum(0,0,0,0,0,tEnd),'HH:MM:SS\n'),...
        'est rem time:',datestr(datenum(0,0,0,0,0,remt),'HH:MM:SS')])
    %fprintf('nM(%d/%d) ',mod(ceil(aaa/Nmethods),nMonte),nMonte)   
    %fprintf('Ep(%d/%d)\n ',ceil(aaa/nMonte/Nmethods),perc/Nmethods/nMonte)
end
















