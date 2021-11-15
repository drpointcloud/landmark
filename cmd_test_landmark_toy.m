
clc
clear
%%

%% Create toy data 1
pz = [10 20 30 40 50];
[px,py]=meshgrid(pz,pz);
centers = [px(:),py(:)];
X_P = @(m) randn(m,2) + centers(randi(size(centers,1),m,1),:);
R = @(ep) chol([1 (ep-1)/(ep+1); (ep-1)/(ep+1) 1]);
Y_P = @(m,ep)repmat([0 0],m,1) + randn(m,2)*R(ep) + centers(randi(size(centers,1),m,1),:);
%C_Q = @(ep) ([1 (ep - 1)/(ep + 1); (ep - 1)/(ep + 1) 1]);
%Y_P = @(m,ep) mvnrnd([0;0],C_Q(ep),m) + centers(randi(size(centers,1),m,1),:);

%% Create toy data 2
X_p = @(m) randn(m,2);
C_Q = @(ep) [1, (ep - 1)/(ep + 1); (ep - 1)/(ep + 1), 1];
Y_p = @(m,ep) mvnrnd([0;0],C_Q(ep),m);%(max(max(X_P(m)))- mean(mean(X_P(m))));
%centers = @(d) cat(1,[0.5,zeros(1,d-1)],[0.0,0.5,zeros(1,d-2)]);
%Y_p = @(m,d) randn(m,d) + sparse(1:m,randi(2,1,m),1,m,2)*centers(d)+4 ;
%%

sample_size = round(100*(linspace(1,5,7))); % Number of samples from p_X
nperms = 100;
eps = linspace(1,12,6);%log(3.^(2:2:40));
nmonte = 250;%10*numel(p_missing);  % must be divisible by 3

if isunix
    slash = '/';
else
    slash = '\';
end
%%
methods = {
    {@(K,x_idx,method) kernel_max_proj_landmark(K,x_idx,method,nperms), 'L-Bures-max',@(x) abs(x)};
    {@(K,x_idx,method) kernel_max_proj_landmark(K,x_idx,method,nperms), 'L-W2-max',@(x) x};
    %{@(K,x_idx,method) kernel_max_proj_landmark(K,x_idx,method,nperms), 'L-Bures-mean',@(x) abs(x)};
    %{@(K,x_idx,method) kernel_max_proj_landmark(K,x_idx,method,nperms), 'L-W2-mean',@(x) x};
    {@(K,x_idx,method) kernel_max_proj(K,x_idx,method,nperms), 'MMD',@(x) x};
    %{@(K,x_idx,method) kernel_max_proj(K,x_idx,method,nperms), 'W2',@(x) x};
    %{@(K,x_idx,method) kernel_max_proj(K,x_idx,method,nperms), 'Bures',@(x) abs(x)};
    %{@(K,x_idx,method) kernel_max_proj(K,x_idx,method,nperms), 'Kolmogorov',@(x) abs(x)};
    %{@(K,x_idx) RuLSIF_kernel(K,x_idx,0),'uLSIF',@(x) x};
    %{@(K,x_idx) RuLSIF_kernel(K,x_idx,0.5),'RuLSIF-0.5',@(x) x};
    %{@(K,x_idx) logReg_kernel(K,x_idx,'z'),'logReg',@(x) x};% requires liblinear-2.11
    %{@max_w2_means,'Max-Sliced W2 (approx.)',@(x) x};
    %{@logReg,'logReg-linear',@(x) x};   % requires liblinear-2.11
    %{@linear_max_sliced_bures,'Max-Sliced Bures',@(x) abs(x)};   % requires minFunc
    %{@linear_max_sliced_w2,'Max-Sliced W2',@(x) x};   % requires minFunc
    };
total_methods = methods;
Nmthds = numel(total_methods);

%%
Time_spnt = zeros(numel(sample_size),numel(eps),nmonte,numel(total_methods));
Div_value = zeros(numel(sample_size),numel(eps),nmonte,numel(total_methods));
Betahat = Div_value;
D_nperms =  zeros(numel(sample_size),numel(eps),nmonte,numel(total_methods),nperms);
%%
aaa = 0;
perc = numel(sample_size)*numel(eps);

timestart = tic;
progress = '..........';
%%
for sample_ii = 1:numel(sample_size)
    N = sample_size(sample_ii);
    
    for eps_ii = 1:numel(eps)
        epsilon = eps(eps_ii);
        
        d_nperms =  zeros(nmonte,numel(total_methods),nperms);
        time_spnt = zeros(nmonte,numel(total_methods));
        betahat = time_spnt;
        div_value = time_spnt;
        aaa=aaa+1;
        
        parfor monte_ii = 1:nmonte
            
            X = X_p(N);
            Y = Y_p(N,epsilon);
            
            Z = double(cat(1,X,Y)); % combine two samples
            x_idx = cat(1,ones(size(X,1),1),zeros(size(Y,1),1))==1; %bool indicator
            [K,~] = gaussian_kernel(Z,'info');
            
            %% tic
            %ssize, p_missing, nmonte,methods, classes, nperms, 2
            %ssize, p_missing, nmonte,methods, classes, 2
            %witness = cell(numel(total_methods),1);
            
            for method_ii = 1:Nmthds %take only max-bures and max-w2 than will add their mean versions
                
                method_way = methods{method_ii}{1};
                method = methods{method_ii}{2};
                tic
                [V,div,alphas,D1] =  method_way(K,x_idx,method);
                
                time_spnt(monte_ii, method_ii) = toc;
                d_nperms(monte_ii, method_ii,:) = D1;
                % if mean(D1>=div), then reject the H_0 (null hypothesis)
                betahat(monte_ii, method_ii) = mean(D1>=div);
                div_value(monte_ii, method_ii) = div;
                %witness{offset+method_ii} = V(x_idx,:);
            end

        end
        D_nperms( sample_ii, eps_ii, :,:,:) = d_nperms;
        Time_spnt(sample_ii, eps_ii, :,:) = time_spnt;
        Betahat(  sample_ii, eps_ii, :,:) = betahat;
        Div_value(sample_ii, eps_ii, :,:) = div_value;

        clc
        progress = progress_print(aaa,perc,timestart,progress);
    end
end

%%
datestr(now)
save('power_toy_1gauss.mat')
Betahatt = squeeze(mean(squeeze(D_nperms)>=Div_value,5));
linestyle= {'-',':','--','-',':','--','-',':','--','-'};
markers = {'*','^','x','s','d'};
colors = {'r','b','k','m','g','y','c'}; %each color for a class: so maximum 7 classes
%methods_name = {'$L^{\ast}$-Bures','$L^{\ast}$-W2','MMD','$\overline{L}$-Bures','$\overline{L}$-W2'};
methods_name = arrayfun(@(i) total_methods{i}{2},(1:Nmthds),'uni',0);
% Betahat  1-ssize, 2-eps,   3-nmonte, 4-methods
%% power curves as a function of epsilon
alpha = [0.05 0.10 0.15];
plot_ssize = sample_size(1:3:end);
n_row = numel(plot_ssize);            % subplot # of rows
n_col = numel(alpha); % # of columns
fig_width = 0.8;fig_height= 0.65;
pad_w = (1-fig_width)/(n_col+1);pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width;axis_h = 1/n_row*fig_height;

figure(1);clf
for j = 1:numel(plot_ssize)
betahat_eps = squeeze(Betahat(end,:,:,:)); %1-eps,2-nmonte,3-methods

for i = 1:numel(alpha)
    alpha_ii = alpha(i); %1-eps,2-methods,3-2.
    k = (j-1)*numel(alpha)+i;
    powhat_eps = squeeze(mean(betahat_eps<alpha_ii,2)); %mean across nmonte
    
    row = floor((k-1)/n_col)+1;col = mod(k-1,n_col)+1;
    axis_l = axis_w*(col-1) + pad_w*col;axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);
    
    kk =0;
    subplot('position', [axis_l axis_b axis_w axis_h] )
    for method_ii=1:numel(total_methods)
        kk=kk+1;
        plot(powhat_eps(:,method_ii),':','linewidth',5,'Marker',markers{kk},'MarkerSize',15);
        hold on
    end
    
    if j==1
        title(['Rejection rate at \alpha  =',sprintf('%0.2f\n',alpha_ii)])
    end
    xlabel(['\epsilon_{(N=',num2str(plot_ssize(j)),')}'])
    xticks(1:numel(eps))
    set(gca,'XTickLabel',sprintf('%0.0f\n',eps(1:end)))
    %ylim([0 1.1])
    legend(methods_name,'FontSize',15,'Location','best','interpreter','latex')
    legend('boxoff')
    set(gca,'xgrid','on','fontsize',18)
    grid on
end
end


%% power curves as a function of sample size
alpha = [0.05 0.10 0.15];
plot_eps = eps(1:2:end);
%to adjust subplot positions
n_row = numel(plot_eps);            % subplot # of rows
n_col = numel(alpha); % # of columns
fig_width = 0.8;fig_height= 0.65;

%to adjust subplot positions
pad_w = (1-fig_width)/(n_col+1);pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width;axis_h = 1/n_row*fig_height;

figure(2),clf
for j = 1:numel(plot_eps)
    betahat_ssize = squeeze(Betahat(:,j,:,:)); %1-ssize 2-nmonte,3-methods
for i = 1:numel(alpha)
    k = (j-1)*numel(alpha)+i;
    alpha_ii = alpha(i);
    powhat_ssize = squeeze(mean(betahat_ssize<alpha_ii,2)); %1-ssize 2-methods,3-2.
    
    %to adjust subplot positions
    row = floor((k-1)/n_col)+1;col = mod(k-1,n_col)+1;
    axis_l = axis_w*(col-1) + pad_w*col;axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);
    
    kk =0;
    subplot('position', [axis_l axis_b axis_w axis_h] )
    for method_ii=1:numel(total_methods)
        kk=kk+1;hold on
        plot(powhat_ssize(:,method_ii),':','linewidth',5,'Marker',markers{kk},'MarkerSize',15);
    end
    if j==1
        title(['Rejection rate at \alpha  =',sprintf('%0.2f\n',alpha_ii)])
    end
    xlabel(['N_{(\epsilon=',sprintf('%.2f%',plot_eps(j)),')}'])
    %ylim([0 1.1])
    xticks(1:numel(sample_size))
    set(gca,'XTickLabel',sprintf('%d\n',sample_size(1:end)))
    legend(methods_name,'FontSize',15,'Location','best','Interpreter','latex');
    legend('boxoff')
    set(gca,'xgrid','on','fontsize',18)
    grid on
end
end


%% computation time as function of sample size
%methods_name = {'$L^{\ast}$-Bures','$L^{\ast}$-W2','MMD'};
methods_name = arrayfun(@(i) total_methods{i}{2},(1:Nmthds),'uni',0);
%time_spnt = 1-ssize,2-eps, 3-nmonte, 4-methods.
comptime_ssize = squeeze(mean(Time_spnt,[2,3]));

figure(3);clf
n_row = 1; % subplot # of rows
n_col = 1;% # of columns
fig_width = 0.8;fig_height= 0.7;

%to adjust subplot positions
pad_w = (1-fig_width)/(n_col+1);pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width;axis_h = 1/n_row*fig_height;
i=n_col;

%to adjust subplot positions
row = floor((i-1)/n_col)+1;col = mod(i-1,n_col)+1;
axis_l = axis_w*(col-1) + pad_w*col;axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);

kk =0;
subplot('position', [axis_l axis_b axis_w axis_h])
for method_ii=1:numel(total_methods)
    kk=kk+1;hold on
    plot(comptime_ssize(:,method_ii),':','linewidth',5,'Marker',markers{kk},'MarkerSize',15);
end

xlabel('Sample size');%,'FontSize',10);%,'FontWeight','Bold')
ylabel('Time (seconds)');
title('Computation time')
xticks(1:numel(sample_size))
set(gca,'XTickLabel',sprintf('%d\n',sample_size(1:end)))
legend(methods_name,'FontSize',15,'Location','southeast','Interpreter','latex');
legend('boxoff')
set(gca,'xgrid','on','fontsize',17)
grid on



%% Plot input data
figure(10),clf

plot_eps = eps([1,2,4]);
neps = numel(plot_eps);
ap = ceil(sqrt(neps)); %to show all plots together as # epsilon varies
aq = ceil(neps/ap);

n_row = 1; % subplot # of rows
n_col = numel(plot_eps);% # of columns
fig_width = 0.8;fig_height= 0.75;

%to adjust subplot positions
pad_w = (1-fig_width)/(n_col+1);
pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width;
axis_h = 1/n_row*fig_height;

m = 10000;
for eps_ii = 1:neps
    epsilon = plot_eps(eps_ii);
    X = X_p(m);
    Y = Y_p(m,epsilon);
    
    %to adjust subplot positions
    i=eps_ii;
    row = floor((i-1)/n_col)+1;
    col = mod(i-1,n_col)+1;
    axis_l = axis_w*(col-1) + pad_w*col;
    axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);
    
    subplot('position', [axis_l axis_b axis_w axis_h])
    
    h=plot(X(:,1),X(:,2),'.','linewidth',2,'MarkerSize',10);
    set(h,'color',[1 .7 .2])
    %set(gca,'xgrid','on','ygrid','on')
    %axis square
    %xlabel('X_1');ylabel('X_2')
    hold on
    
    h = plot(Y(:,1),Y(:,2),'.','linewidth',5,'MarkerSize',15);
    set(h,'color',[.4 1 1])
    %set(gca,'xgrid','on','ygrid','on')
    %axis square
    %xlabel('Y_1');ylabel('Y_2')
    title(['\epsilon=',num2str(round(epsilon))])
    set(gca,'xgrid','on','ygrid','on','fontsize',15)
    grid on
    legend('X','Y')
    %legend('boxoff')
end

% i=4;
% row = floor((i-1)/n_col)+1;
% col = mod(i-1,n_col)+1;
% axis_l = axis_w*(col-1) + pad_w*col;
% axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);
% 
% kk =0;
% subplot('position', [axis_l axis_b axis_w axis_h])
% for method_ii=1:numel(total_methods)
%     kk=kk+1;hold on
%     plot(comptime_ssize(:,method_ii),':','linewidth',5,'Marker',markers{kk},'MarkerSize',15);
% end
% xlabel('Sample size');%,'FontSize',10);%,'FontWeight','Bold')
% ylabel('Time (seconds)');
% title('Computation time')
% xticks(1:numel(sample_size))
% set(gca,'XTickLabel',sprintf('%d\n',sample_size(1:end)))
% legend(methods_name,'FontSize',15,'Location','best','Interpreter','latex');
% legend('boxoff')
% set(gca,'xgrid','on','fontsize',17)
% grid on
%% power-contour plots: epsilon-sample size
alpha = [0.05 0.10 0.15];
figure(30);clf

n_row = 3;            % subplot # of rows
n_col = numel(alpha); % # of columns
fig_width = 0.8;fig_height= 0.75;

%to adjust subplot positions
pad_w = (1-fig_width)/(n_col+1);pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width;axis_h = 1/n_row*fig_height;
  
[X, Y] = meshgrid(eps,sample_size);

powhat_eps_ssize = zeros(numel(sample_size),numel(eps),numel(methods));
for i = 1:numel(alpha)
    alpha_ii = alpha(i); %1-eps,2-methods,3-2.
    
    powhat_eps_ssize = squeeze(mean(Betahat<alpha_ii,3));
    
    kk =0;
    for j=1:numel(total_methods)
        kk=kk+1;
        k = (j-1)*numel(alpha)+i;
        %to adjust subplot positions
        row = floor((k-1)/n_col)+1;col = mod(k-1,n_col)+1;
        axis_l = axis_w*(col-1) + pad_w*col;axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);
        
        subplot('position', [axis_l axis_b axis_w axis_h] )
        [M,c] = contourf(X,Y,powhat_eps_ssize(:,:,j),colors{kk});
        c.LineWidth = 2;
        c.LineStyle = linestyle{kk};
        
        if j==1
            title(['Rejection rate at \alpha  =',sprintf('%0.2f\n',alpha_ii)])
        end
        if i==1
            methods_name = total_methods{j}{2};
            xlabel('P-missing','FontSize',10,'FontWeight','Bold')
            ylabel(['Sample size - ',sprintf('%s',methods_name)] ,'FontSize',10,'FontWeight','Bold')
            
        end
    end
end


%% Box plots
figure(31),clf

methods_name = arrayfun(@(i) total_methods{i}{2},(1:3),'uni',0);
divs = Div_value;                     % 1-ssize, 2-eps, 3-nmonte,4-methods
divspC = squeeze(mean(divs,[1 2]));   % 1nmonte, 2methods
divsMC = squeeze(mean(divs,[1 3]));   % 1eps, 2methods
divsMpC = squeeze(mean(divspC,1));    % 1methods

divsep = squeeze(mean(divs,2));    % mean: eps
divsS = squeeze(mean(divs,1));    % mesn:ssize

plotsize = min(numel(eps),numel(sample_size));
plotdata = divsS;
for j = 1:2    
    if j==2
        plotdata = divsep;
    end
for i = 1:plotsize

    subplot(2,plotsize,i+(j-1)*plotsize)
    boxplot(squeeze(plotdata(i,:,:)),{methods_name})
end
end

bxplt_nMont = divspC + (1-divsMpC);
bxplt_eps = divsMC + (1-divsMpC);
p_eps_normalized = (bxplt_eps - min(bxplt_eps)) ./ (max(bxplt_eps) - min(bxplt_eps));

if size(Betahat,1)<5
    betahat_eps = squeeze(mean(Betahat,1)); %betahat_pmiss(eps,nmonte,methods)
    powhat_eps = squeeze(mean(betahat_eps<alpha_ii,2)); %powhat_pmiss(pmiss,methods)
end

figure(10),clf
subplot(321); boxplot(bxplt_eps,{methods_name})
subplot(322); boxplot(p_eps_normalized,{methods_name})
subplot(323); semilogy(powhat_eps,'linewidth',3)
subplot(324); semilogy(1-powhat_eps,'linewidth',3);

subplot(325); boxplot(bxplt_nMont,{methods_name})
xlabel('(X\sim\mu,Y\sim\nu) are drawn for 250 times')
ylabel('Divergence variation')

subplot(326); boxplot(bxplt_eps,{methods_name})
xlabel(['Divergence variations for \epsilon\in[',num2str(min(eps)),',',num2str(max(eps)),']'])
ylabel('Divergence value')
ylim([0 1.1])

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
disp(['elapsed time:',datestr(datenum(0,0,0,0,0,tEnd),'DD:HH:MM:SS\n'),...
    'est rem time:',datestr(datenum(0,0,0,0,0,remt),'DD:HH:MM:SS')])
%fprintf('nM(%d/%d) ',mod(ceil(aaa/Nmethods),nMonte),nMonte)
%fprintf('Ep(%d/%d)\n ',ceil(aaa/nMonte/Nmethods),perc/Nmethods/nMonte)
end
















