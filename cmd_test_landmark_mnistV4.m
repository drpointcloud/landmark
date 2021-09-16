
clc
clear
load('/Users/yuksel/Documents/Dataset/EMNIST/emnist-mnist.mat');
%rng(10);  % set the random seed

% Requires minFunc_2012
%  addpath(genpath('path_to/minFunc_2012'))
%%
N = 400;%N = 100;%%%% N = 500; % Number of samples from p_X
nperms = 100;
p_missing = linspace(0.05,1,8);% for scenario two % the digit has 100*(1-q)% of
nmonte = 250;%10*numel(p_missing);  % must be divisible by 3
classes = ['0','1','2','3'];%,'4','5','6','7','8','9'];

%%
method1 = {
    {@(K,x_idx) L_MSKB_one_side_minfunc(K,x_idx,nperms), 'L-Bures',@(x) abs(x)};
    %{@(K,x_idx) L_MSKW_minfunc(K,x_idx,nperms), 'L-W2',@(x) x};
    };
method2 = {
    %{@(K,x_idx,method) kernel_max_proj(K,x_idx,method,nperms), 'W2',@(x) x};
    %{@(K,x_idx,method) kernel_max_proj(K,x_idx,method,nperms), 'Bures',@(x) abs(x)};
    {@(K,x_idx,method) kernel_max_proj(K,x_idx,method,nperms), 'MMD',@(x) x};
    %{@(K,x_idx,method) kernel_max_proj(K,x_idx,method,nperms), 'Kolmogorov',@(x) abs(x)};
    };
add_methods = {
    %{@(K,x_idx) RuLSIF_kernel(K,x_idx,0),'uLSIF',@(x) x};
    %{@(K,x_idx) RuLSIF_kernel(K,x_idx,0.5),'RuLSIF-0.5',@(x) x};
    %{@(K,x_idx) logReg_kernel(K,x_idx,'z'),'logReg',@(x) x};% requires liblinear-2.11
    };
lin_methods = {
    %{@max_w2_means,'Max-Sliced W2 (approx.)',@(x) x};
    %{@logReg,'logReg-linear',@(x) x};   % requires liblinear-2.11
    %{@linear_max_sliced_bures,'Max-Sliced Bures',@(x) abs(x)};   % requires minFunc
    %{@linear_max_sliced_w2,'Max-Sliced W2',@(x) x};   % requires minFunc
    };
total_methods = cat(1,method1,method2,add_methods,lin_methods);
Nmthds = numel(total_methods);
Npmiss = numel(p_missing);

%%
time_spent = nan(numel(p_missing),nmonte,numel(total_methods),numel(classes));
auc = time_spent;
p_at_10 = auc;
obj_val = p_at_10;
D = obj_val;
d = zeros(nperms,numel(p_missing),nmonte,numel(total_methods),numel(classes));
landmark = zeros(numel(p_missing),nmonte,numel(total_methods),numel(classes));
betahat = landmark;
orderings = cell(size(landmark));
%%
aaa = 0;
perc = numel(p_missing)*nmonte*numel(total_methods)*numel(classes);
timestart = tic;
progress = '..........'; 
%%
for p_ii = 1:numel(p_missing)
    p_with_missing = p_missing(p_ii);
    
    for monte_ii = 1:nmonte
        %if mod(monte_ii-1,3)+1==2
         %   p_ii = p_ii + 1 ;
        %end
        
        for l = classes%['0','1','2','3','4','5','6','7','8','9']
            %scenario = mod(monte_ii-1,3)+1;
            scenario =2;
            switch scenario
                case 1  % equivalant to outlier detection
                    %https://www.jmlr.org/papers/volume10/kanamori09a/kanamori09a.pdf
                    % X has all characters
                    % Y has one missing
                    % density ratio p_Y(x)/p_X(x) should be very small when x is missing
                    x_sel = randperm(numel(dataset.train.labels),N);
                    not_label = dataset.mapping(dataset.mapping(:,2)==l,1);
                    subset = find(dataset.test.labels~=not_label);
                    y_sel = subset(randperm(numel(subset),N));
                    
                case 2
                    %p_with_missing = p_missing(p_ii);
                    % X has very few of missing
                    % Y has all characters
                    % density ratio p_Y(x)/p_X(x) should be very large for x in missing
                    y_sel = randperm(numel(dataset.test.labels),N);
                    not_label = dataset.mapping(dataset.mapping(:,2)==l,1);
                    subset = find(dataset.train.labels~=not_label);
                    x_sel = subset(randperm(numel(subset),round(N*p_with_missing)));
                    x_sel2 = randperm(numel(dataset.train.labels), N - round(N*p_with_missing));
                    x_sel = cat(1,x_sel(:),x_sel2(:));
                    %[a,~]=hist(L_X,unique(dataset.train.labels(x_sel)))
                    
                case 3
                    % X has all characters
                    % Y has only the selected character
                    % density ratio p_Y(x)/p_X(x) should be large for x is selected
                    x_sel = randperm(numel(dataset.train.labels),N);
                    not_label = dataset.mapping(dataset.mapping(:,2)==l,1);
                    subset = find(dataset.test.labels==not_label);
                    y_sel = subset(randperm(numel(subset),N));
            end
            
            X = dataset.train.images(x_sel,:);
            L_X = dataset.train.labels(x_sel);
            
            Y = dataset.test.images(y_sel,:);
            L_Y = dataset.test.labels(y_sel);
            
            %%
            Z = double(cat(1,X,Y)); % combine two samples
            x_idx = cat(1,ones(size(X,1),1),zeros(size(Y,1),1))==1; %bool indicator
            
            [K,sigma] = gaussian_kernel(Z,'median');
            kappa = @(x,y,sigma) exp(-( sum(x.^2,2) + sum(y.^2,2).' -2*x*y.')/2/sigma^2);
            
            %% method1
            tic
            %1nperms,2p_missing,3nmonte,4methods,5classes
            %        1p_missing,2nmonte,3methods,4classes
            witness = cell(numel(total_methods),1);
            offset =0;
            for method_ii = 1:numel(method1)
                aaa=aaa+1;
                tic
                method = method1{method_ii}{1};
                [V,div,alphas,D1] =  method(K,x_idx);
                
                [~,idx]= max(div);
                V = V(:,idx);
                witness{method_ii} = V(x_idx,:);
                
                time_spent( p_ii,monte_ii,offset+method_ii,not_label+1) = toc;
                obj_val(    p_ii,monte_ii,offset+method_ii,not_label+1) = div(idx);
                d(:,        p_ii,monte_ii,offset+method_ii,not_label+1) = D1;
                D(          p_ii,monte_ii,offset+method_ii,not_label+1) = mean(D1<=div);
                betahat(    p_ii,monte_ii,offset+method_ii,not_label+1) = mean(D1>=div);
            end
            
            %% method2
            offset = offset+numel(method1);
            for method_ii = 1:numel(method2)
                aaa=aaa+1;
                method_way = method2{method_ii}{1};
                method = method2{method_ii}{2};
                tic
                [V,div,alphas,D1] =  method_way(K,x_idx,method);
                
                [~,idx]= max(div);
                V = V(:,idx);
                witness{offset+method_ii} = V(x_idx,:);
                
                time_spent( p_ii,monte_ii,offset+method_ii,not_label+1) = toc;
                obj_val(    p_ii,monte_ii,offset+method_ii,not_label+1) = div(idx);
                d(:,        p_ii,monte_ii,offset+method_ii,not_label+1) = D1;
                D(          p_ii,monte_ii,offset+method_ii,not_label+1) = mean(D1<=div);
                betahat(    p_ii,monte_ii,offset+method_ii,not_label+1) = mean(D1>=div);
            end
            
            %% add_methods
            offset = offset+numel(method2);
            for method_ii = 1: numel(add_methods)
                aaa=aaa+1;
                tic
                beta=add_methods{method_ii}{1}(K,x_idx);
                
                time_spent(offset+method_ii,not_label+1,monte_ii) = toc;
                witness{offset+method_ii} = beta;
            end
            
            %% Linear methods
            offset = offset + numel(add_methods);
            for method_ii = 1: numel(lin_methods)
                aaa=aaa+1;
                tic
                w_z =lin_methods{method_ii}{1}(Z,x_idx);
                w_x = w_z(x_idx);
                w_y = w_z(~x_idx);
                time_spent(offset+method_ii,not_label+1,monte_ii) = toc;
                witness{offset+method_ii} = w_x;
            end
            toc
            clc
            progress = progress_print(aaa,perc,timestart,progress);
            %%
            aucs = zeros(numel(total_methods),1);
            kk = 0;
            for method_ii =  1:numel(total_methods)
                kk= kk +1;
                % default plotting
                values = total_methods{method_ii}{3}(witness{method_ii});
                [~,ordering] = sort(values,'descend');
                
                auc1 = calcAUC(values,L_X==not_label);
                auc2 = calcAUC(-values,L_X==not_label);
                
                %disp([auc1 auc2])
                if auc1>auc2
                    aucs(method_ii) = auc1;
                else
                    aucs(method_ii) = auc2;
                    ordering = ordering(end:-1:1);
                end
                p_at_10(p_ii,monte_ii,method_ii,not_label+1) = mean(L_X(ordering(1:numel(classes)))==not_label);
                orderings{p_ii,monte_ii,method_ii,not_label+1} = ordering;
            end
            auc(p_ii,monte_ii,:,not_label+1) = aucs;
        end
    end
end

%%
% auc = nan(total_methods,classes,nmonte,p_missing);
M_mean_auc1 = cell2mat(arrayfun(@(i) mean(reshape(auc(:,:,i:3:end),Nmthds,[]),2,'omitnan'),(1:3),'uni',0));
M_mean_auc2 = cell2mat(arrayfun(@(i)      reshape(auc(:,:,i:3:end),Nmthds,[]),             (1:3),'uni',0));
M_std_auc = cell2mat(arrayfun(@(i) std(reshape(auc(:,:,i:3:end), Nmthds,[]),0,2,'omitnan'),(1:3),'uni',0));
cellfun(@(x) x{2},total_methods,'uni',0);
M_mean_auc = mean(M_mean_auc2,2,'omitnan');

% win rate per task
M = reshape(auc,numel(total_methods),numel(classes),3,[]);
M1 = cell2mat(arrayfun(@(i) sum(sum(M(:,:,i,:) >= max(M(:,:,i,:),[],1) ,2),4),(1:3),'uni',0));
M2 = squeeze(mean(squeeze(M(:,:,2,:)),2,'omitnan'));

%save('mnist_runs','p_at_10','auc','time_spent','N','methods','total_methods','p_with_missing')
linestyle= {'-',':','--','-',':','--','-',':','--','-'};


%% Plot power curves
% 1p_missing,2nmonte,3methods,4classes
% D0 = squeeze(mean(D,4));
% D1 = arrayfun(@(i)reshape(D0(:,i:3:end,:),numel(p_missing),[],Nmthds),(1:3),'uni',0);
% D2 = permute(cell2mat(permute(D1,[1,3,2])),[1,3,2]);
% nan(numel(total_methods),numel(classes),nmonte);
% D3 = squeeze(mean(D,2));
% D4 = arrayfun(@(i)reshape(D3(:,i:3:end),Nmthds,[]),(1:3),'uni',0);
% D5 = permute(cell2mat(permute(D4,[1,3,2])),[1,3,2]);
alpha = [0.05 0.10 0.15];
betaj = squeeze(mean(betahat,4));
betai = squeeze(mean(D,4));
figure(3);clf
for i = 1:numel(alpha)
    alpha_ii = alpha(i);
    % cv0 = squeeze(quantile(obj,alpha,2));
    % cv1 = squeeze(quantile(obj,1-alpha));
    powhat = squeeze(mean(betaj<alpha_ii,2));
    % DD = dq;
    % D0 = squeeze(mean(DD,2));
    % D1 = arrayfun(@(i)reshape(D0(:,i:3:end),Nmthds,[]),(1:3),'uni',0);
    % D2 = permute(cell2mat(permute(D1,[1,3,2])),[1,3,2]);    
    subplot(1,3,i)
    semilogy(1-powhat,'linewidth',3);
    set(gca,'fontsize',10)
    set(gca,'xgrid','on')
    title(['Rejection rate (power) at \alpha  =',sprintf('%0.2f\n',alpha_ii)])
    xlabel('p-missing','FontSize',10);%,'FontWeight','Bold')
    xticks(1:numel(p_missing))
    set(gca,'XTickLabel',sprintf('%0.1f\n',p_missing))
    ylim([0 1.1])
%     title(['power test of toy data \alpha =',num2str(alpha_ii)])
    legend('L-Bures','MMD','Location','southeast')%,'Unbiased MMD')
end


%%
%% Box plots
%methods = {'L-Bures','L-W2','MMD'};
divs = obj_val;                     % 1p_missing,2nmonte,3methods,4classes
divsp = squeeze(mean(divs,1));     % mean along p_miss:   1nmonte, 2methods,3classes
divspC = squeeze(mean(divsp,3));   % mean along classes:  1nmonte, 2methods
divsM = squeeze(mean(divs,2));     % mean along Nmonte:   1p_miss, 2methods,3classes
divsMC =  squeeze(mean(divsM,3));  % mean along classes:  1p_miss, 2methods
divsMpC = squeeze(mean(divspC,2)); % mean along p_miss:   1methods

bxplt_nMont = zeros(size(divs,2),numel(total_methods));
bxplt_pmiss = zeros(size(divs,1),numel(total_methods));
methods_name = cell(1,numel(total_methods));



for method_ii=1:numel(total_methods)
    bxplt_nMont(:,method_ii) = divspC(:,method_ii) + (1-divsMpC(method_ii));
    bxplt_pmiss(:,method_ii) = divsMC(:,   method_ii) + (1-divsMpC(method_ii));
    
    methods_name{method_ii} = total_methods{method_ii}{2};
end

figure(10),clf
bxplt_pmissing = (bxplt_pmiss - min(bxplt_pmiss)) ./ (max(bxplt_pmiss) - min(bxplt_pmiss));

subplot(2,2,1)
boxplot(bxplt_pmiss,{methods_name})

subplot(2,2,2)
boxplot(bxplt_pmissing,{methods_name})

subplot(2,2,3)
semilogy(1-bxplt_pmissing,'linewidth',3);

betaj = squeeze(mean(betahat,4));
powhat = squeeze(mean(betaj<alpha_ii,2));
subplot(2,2,4)
semilogy(1-powhat,'linewidth',3);


% title('Divergence variations for 250 draws')
figure(11),clf
subplot(121)
boxplot(bxplt_nMont,{methods_name})
xlabel('(X\sim\mu,Y\sim\nu) are drawn for 250 times')
ylabel('Divergence variation')
%as a function of epsilon
subplot(122)
boxplot(bxplt_pmiss,{methods_name})
xlabel(['Divergence variations for \epsilon\in[',num2str(min(eps)),',',num2str(max(eps)),']'])
ylabel('Divergence value')


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
        'est rem time:',datestr(datenum(0,0,0,0,0,remt),'DD:HH:MM:SS')])
    %fprintf('nM(%d/%d) ',mod(ceil(aaa/Nmethods),nMonte),nMonte)   
    %fprintf('Ep(%d/%d)\n ',ceil(aaa/nMonte/Nmethods),perc/Nmethods/nMonte)
end

















%%
