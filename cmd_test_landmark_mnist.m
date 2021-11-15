
clc
clear
% format short
%load('/Users/yuksel/Documents/Dataset/FashionMNIST/fashion_mnist.mat')
%load('/Users/yuksel/Documents/Dataset/CIFAR/cifar_inception_2048.mat');
%load('/Users/yuksel/Documents/Dataset/FashionMNIST/fashion_MNIST_inception_2048.mat');
%load('C:\Users\yuksel\Google Drive\Resources\Implementation\MATLAB\Landmark_clean\emnist-mnist.mat');
load('/Users/yuksel/Documents/Dataset/EMNIST/emnist-mnist.mat');

%rng(10);  % set the random seed
%%
% Requires minFunc_2012
%  addpath(genpath('path_to/minFunc_2012'))
%%
sample_size = 500;%round(100*linspace(1,7,5)); % Number of samples from p_X
nperms = 1;
p_missing = 0.85;%linspace(0,1,9);% for scenario two % the digit has 100*(1-q)% of
nmonte = 1;%10*numel(p_missing);  % must be divisible by 3
classes = ['0','1','2','3','4','5','6','7','8','9'];

%%
methods = {
    {@(K,x_idx,method) kernel_max_proj_landmark(K,x_idx,method,nperms), 'L-Bures-max',@(x) x};
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
Npmiss = numel(p_missing);

%%
D_nperms =  zeros(numel(sample_size),numel(p_missing),numel(classes),nmonte,Nmthds,nperms);
Time_spnt = nan(numel(sample_size),numel(p_missing),numel(classes),nmonte,Nmthds);
Div_value = Time_spnt;
Betahat = Div_value;
Witness = cell(size(Time_spnt));
Alphas = Witness;
Auc = nan(numel(sample_size),numel(p_missing),numel(classes),nmonte,Nmthds);
P_at_10 = Auc;
Orderings = cell(size(Auc));

%%
aaa = 0;
perc = numel(sample_size)*numel(p_missing)*numel(classes);
timestart = tic;
progress = '..........';
%%
for sample_ii = 1:numel(sample_size)
    N = sample_size(sample_ii);
    
    for pmiss_ii = 1:numel(p_missing)
        p_with_missing = p_missing(pmiss_ii);
        
        for l = classes%['0','1','2','3','4','5','6','7','8','9']
            label_ii = dataset.mapping(dataset.mapping(:,2)==l,1)+1;
            
            %h = figure(100*sample_ii + 10*pmiss_ii+label_ii-1);clf
            %set(h,'name',['digit',sprintf('%d',label_ii-1)],'numbertitle','on')%name of the figure
            %set(h,'WindowStyle','docked') %dock the figure
            
            %d_nperms =  zeros(nmonte,Nmthds,nperms);
            time_spnt = zeros(nmonte,Nmthds);
            %div_value = time_spnt;
            %betahat = div_value;
            witness = cell(size(time_spnt));
            %alphas = witness;
            auc = nan(nmonte,Nmthds);
            p_at_10 = auc;
            %orderings = cell(size(auc));          
            
            aaa=aaa+1;
            for monte_ii = 1:nmonte
                
                %X has very few of missing %Y has all characters
                %density ratio p_Y(x)/p_X(x) should be very large for x in missing
                y_sel = randperm(numel(dataset.test.labels),N);
                not_label = dataset.mapping(dataset.mapping(:,2)==l,1);
                subset = find(dataset.train.labels~=not_label);
                x_sel1 = subset(randperm(numel(subset),round(N*p_with_missing)));
                x_sel2 = randperm(numel(dataset.train.labels), N - round(N*p_with_missing));
                x_sel = cat(1,x_sel1(:),x_sel2(:));
                %[a,~]=hist(L_X,unique(dataset.train.labels(x_sel)))
                
                X = dataset.train.images(x_sel,:);
                L_X = dataset.train.labels(x_sel);
                Y = dataset.test.images(y_sel,:);
                L_Y = dataset.test.labels(y_sel);
                
                %%
                Z = double(cat(1,X,Y)); % combine two samples
                x_idx = cat(1,ones(size(X,1),1),zeros(size(Y,1),1))==1; %bool indicator
                [K,sigma] = gaussian_kernel(Z,'median');
                
                %% methods
                for method_ii = 1:Nmthds 
                    
                    method_way = methods{method_ii}{1};
                    method = methods{method_ii}{2};
                    %tic
                    [V,~,~,~] =  method_way(K,x_idx,method);
                    %[V,divs,alpha,D1] =  method_way(K,x_idx,method);
                    %time_spnt(monte_ii, method_ii) = toc;
                    %d_nperms(monte_ii, method_ii,:) = D1;
                    % if mean(D1>=div), then reject the H_0 (null hypothesis)
                    %betahat(monte_ii, method_ii) = mean(D1>=divs);
                    %div_value(monte_ii, method_ii) = divs;
                    witness{monte_ii,method_ii} = V;
                    %alphas{monte_ii,method_ii} = alpha;
                %end
                
                %%
                %for method_ii = 1:Nmthds

                    values = witness{monte_ii,method_ii}(~x_idx); %## works
                    [~,ordering] = sort(values,'descend');
                    
                    auc1a = calcAUC(values,L_Y==not_label);
                    auc1b = calcAUC(-values,L_Y==not_label);
                    
                    if auc1a>auc1b
                        auc(monte_ii,method_ii) = auc1a;
                    else
                        auc(monte_ii,method_ii) = auc1b;
                        ordering = ordering(end:-1:1);
                    end
                    p_at_10(monte_ii,method_ii) = mean(L_Y(ordering(1:numel(classes)))==not_label); %## works
                    %orderings{monte_ii,method_ii} = ordering;
                    %%
%                     k=10;
%                     to_display = ordering(1:k)';
%                     % mat2cell(Imat_normalize,32,32,3,ones(256,1));
%                     % recombobulate = @(M) mat2cell(permute(reshape(M,[],32,32,3),[2 3 4 1]),32,32,3,ones(size(M,1),1));
%                     recombobulate = @(M) mat2cell(permute(reshape(M,[],28,28,1),  [2 3 1]),  28,28,  ones(size(M,1),1));
%                     add_border = @(I) cat(2,ones(30,1),  cat(1,ones(1,29),  cat(2,cat(1,I,ones(1,28)),  ones(29,1))));
%                     % add_border = @(I) cat(2,ones(34,1,3),cat(1,ones(1,33,3),cat(2,cat(1,I,ones(1,32,3)),ones(33,1,3))));
%                     img_mat10 = @(Icell) cell2mat(reshape(cellfun(@(x) add_border(x),Icell,'uni',0),1,[]));
%                     img_disp = @(M) img_mat10(recombobulate(M));
%                     
%                     subplot(numel(methods)+1,1,method_ii)
%                     I_out =img_disp(Y(to_display,:));
%                     %bbb = recombobulate(I(to_display,:));
%                     %ccc = img_mat10(bbb);
%                     %I_out =  cat(2,bbb{:});
%                     imagesc(I_out)
%                     %axis equal
%                     %axis off
%                     %set(gca,'visible','off')
%                     %imwrite(I_out,filenames{do_twice});
%                     colormap(gray)
%                     method_name = methods{method_ii}{2};
%                     title(sprintf('%s',method_name))
                    
                end  %methods
            end% nmonte
            %1-ssize,2-pmiss,3-class,4-nmonte,5-methods,6-2
            %Time_spnt(sample_ii, p_miss_ii,label_ii,:,:) = time_spnt;
            %D_nperms( sample_ii, p_miss_ii,label_ii,:,:,:) = d_nperms;
            %Betahat(  sample_ii, p_miss_ii,label_ii,:,:) = betahat;
            %Div_value(sample_ii, p_miss_ii,label_ii,:,:) = div_value;
            %Witness(sample_ii, pmiss_ii,label_ii,:,:) = witness;
            %Alphas(sample_ii, p_miss_ii,label_ii,:,:) = alphas;
            %Auc(    sample_ii, pmiss_ii,label_ii,:,:) = auc;
            P_at_10(sample_ii, pmiss_ii,label_ii,:,:) = p_at_10;
            %Orderings(sample_ii, p_miss_ii,label_ii,:,:) = orderings;
            
            clc
            progress = progress_print(aaa,perc,timestart,progress);
        end % classes
    end % p_missing
end % sample size
%%
datestr(now)
linestyle= {':',':',':','-','-','-','-',':','--','-'};
markers = {'*','^','x','s','d'};
colors = {'m','b','g','r','k','y','c'}; %each color for a class: so maximum 7 classes
fontsize = 14;
linewidth = 3;
markersize = 10;
legendfontsize = 16;
%save('power_mnist_full_LY.mat')
%% power as a function of p-missing
methods_name = arrayfun(@(i) total_methods{i}{2},(1:Nmthds),'uni',0);
pow_alpha = [0.05 0.10 0.15];

%betahat_pmiss(ssize,p_miss,class,nmonte,methods)
betahat_pmiss = Betahat(end,:,1,:,:);

%to adjust subplot positions: do not change
n_row = 1; n_col = numel(pow_alpha); % # of columns and rows
fig_width = 0.8;fig_height= 0.65;
pad_w = (1-fig_width)/(n_col+1); pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width; axis_h = 1/n_row*fig_height;

figure(1);clf
for i = 1:numel(pow_alpha)
    alpha_ii = pow_alpha(i);
    powhat_pmiss = mean(betahat_pmiss<alpha_ii,4); %mean across nmonte
    
    %to adjust subplot positions
    row = floor((i-1)/n_col)+1; col = mod(i-1,n_col)+1;
    axis_l = axis_w*(col-1) + pad_w*col; axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);
    
    kk =0;
    subplot('position', [axis_l axis_b axis_w axis_h] )
    for method_ii=1:numel(total_methods)
        kk=kk+1;
        plot(powhat_pmiss(:,:,:,:,method_ii),':','linewidth',linewidth,...
            'Marker',markers{kk},'MarkerSize',markersize); hold on
    end    
    
    t = title(['Rejection rate at \alpha  =',sprintf('%0.2f\n',alpha_ii)]);
    set(t,'position',get(t,'position')-[0 axis_b/10 0])
    xlabel('Prevalence of underrepresented digit');
    xticks(1:2:numel(p_missing))
    set(gca,'XTickLabel',sprintf('%0.2f\n',(1-p_missing(1:2:end))/10),'fontsize',fontsize)
    legend(methods_name,'FontSize',legendfontsize,'Location','southeast','interpreter','latex');legend('boxoff')
    grid on
end

%% power as a function of sample size
% betahat_ssize(samp_sz,pmiss,class,nmonte,methods)
betahat_ssize = Betahat(:,1,1,:,:); % choose a p_missing value

%to adjust subplot positions: do not change
n_row = 1; n_col = numel(pow_alpha); % # of columns and rows
fig_width = 0.8; fig_height= 0.65;
pad_w = (1-fig_width)/(n_col+1); pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width; axis_h = 1/n_row*fig_height;

figure(2),clf
for i = 1:numel(pow_alpha)
    alpha_ii = pow_alpha(i);
    powhat_ssize = mean(betahat_ssize<alpha_ii,4);
    
    %to adjust subplot positions
    row = floor((i-1)/n_col)+1; col = mod(i-1,n_col)+1;
    axis_l = axis_w*(col-1) + pad_w*col; axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);
    
    kk =0;
    subplot('position', [axis_l axis_b axis_w axis_h] )
    for method_ii=1:numel(total_methods)
        kk=kk+1;
        plot(powhat_ssize(:,:,:,:,method_ii),':','linewidth',linewidth,...
            'Marker',markers{kk},'MarkerSize',markersize);hold on
    end

    t = title(['Rejection rate at \alpha  =',sprintf('%0.2f\n',alpha_ii)]);
    set(t,'position',get(t,'position')-[0 axis_b/10 0])
    xlabel('Sample size (N)');
    xticks(1:2:numel(sample_size))
    set(gca,'XTickLabel',sprintf('%d\n',sample_size(1:2:end)),'fontsize',fontsize)
    legend(methods_name,'FontSize',legendfontsize,'Location','southeast','Interpreter','latex');legend('boxoff')
    grid on
end

%% plot AUC as a function of sample size
% 1-ssize, 2-pmiss, 3-classes,4-nmonte, 5-methods
mean_auc = mean(Auc(:,:,1,:,:),4);

%to adjust subplot positions: do not change
n_row = 1; n_col = 2; % # of columns and rows
fig_width = 0.8;fig_height= 0.65;
pad_w = (1-fig_width)/(n_col+1); pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width; axis_h = 1/n_row*fig_height;

figure(20),clf
for i=1:n_col
    %to adjust subplot positions
    row = floor((i-1)/n_col)+1;col = mod(i-1,n_col)+1;
    axis_l = axis_w*(col-1) + pad_w*col;axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);
    if i==1
        subplot('position', [axis_l axis_b axis_w axis_h] )
        kk=0;
        for method_ii=1:numel(total_methods)
            kk=kk+1;
            plot(mean_auc(end,:,:,:,method_ii),'linestyle',linestyle{kk},...
                'linewidth',linewidth,'Marker',markers{kk},'MarkerSize',markersize);hold on
        end
        
        t = title(['AUC (N=',sprintf('%d)\n',sample_ii)]);%,'Position',[])
        set(t,'position',get(t,'position')-[0 axis_b/50 0])
        xlabel('Prevalence of underrepresented class');ylabel('AUC');
        xticks(1:numel(p_missing)-1)
        set(gca,'XTickLabel',sprintf('%0.3f\n',(1-p_missing(2:end))/10),'fontsize',fontsize)
        legend(methods_name,'FontSize',legendfontsize,'Location','southeast','interpreter','latex')
        grid on
        
    elseif i==2
        kk=0;
        p_miss = 1;
        subplot('position', [axis_l axis_b axis_w axis_h])
        for method_ii=1:numel(total_methods)
            kk=kk+1;
            plot(mean_auc(:,p_miss,:,:,method_ii),'linestyle',linestyle{kk},...
                'linewidth',linewidth,'Marker',markers{kk},'MarkerSize',markersize);hold on
        end
        
        t = title(['AUC (Prevalence = ',sprintf('%0.3f)\n',(1-p_missing(p_miss))/10)]);
        set(t,'position',get(t,'position')-[0 axis_b/50 0])
        xlabel('Sample size (N)'); ylabel('AUC'); 
        xticks(1:numel(sample_size))
        set(gca,'XTickLabel',sprintf('%d\n',sample_size),'fontsize',fontsize)
        legend(methods_name,'FontSize',legendfontsize,'Location','southeast','interpreter','latex')
        grid on
    else
    end
end


%% plot p@10 as a function of sample size        
methods_name = arrayfun(@(i) total_methods{i}{2},(1:3),'uni',0);
% 1-ssize, 2-pmiss, 3-classes,4-nmonte, 5-methods
mean_p10 = mean(P_at_10(:,:,1,:,:),4);

%to adjust subplot positions: do not change
n_row = 1; n_col = 2; % # of columns and rows
fig_width = 0.8;fig_height= 0.65;
pad_w = (1-fig_width)/(n_col+1); pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width; axis_h = 1/n_row*fig_height;

%x_axis:p_miss
i=1; %to adjust subplot positions
row = floor((i-1)/n_col)+1;col = mod(i-1,n_col)+1;
axis_l = axis_w*(col-1) + pad_w*col;axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);

figure(21),clf
kk=0;
subplot('position', [axis_l axis_b axis_w axis_h] )
for method_ii=1:numel(total_methods)
    kk=kk+1;
    plot(mean_p10(end,:,:,:,method_ii),'linestyle',linestyle{kk},...
        'linewidth',3,'Marker',markers{kk},'MarkerSize',10);hold on
end

%ylim([0 1])
t = title(['Precision@10 (N=',sprintf('%d)\n',sample_size(end))]);
set(t,'position',get(t,'position')-[0 axis_b/50 0])
xlabel('Prevalence of underrepresented class'); %ylabel('Precision at 10');
xticks(1:numel(p_missing))
set(gca,'XTickLabel',sprintf('%0.3f\n',(1-p_missing(1:end))/10),'fontsize',fontsize)
legend(methods_name,'FontSize',legendfontsize,'Location','southeast','interpreter','latex');legend('boxoff')
grid on

%x_axis:sample size
i=2; %to adjust subplot positions
row = floor((i-1)/n_col)+1;col = mod(i-1,n_col)+1;
axis_l = axis_w*(col-1) + pad_w*col;axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);

kk=0;
p_miss = 1;
subplot('position', [axis_l axis_b axis_w axis_h] )
for method_ii=1:numel(total_methods)
    kk=kk+1;
    plot(mean_p10(:,p_miss,method_ii),'linestyle',linestyle{kk},...
        'linewidth',linewidth,'Marker',markers{kk},'MarkerSize',10);hold on
end

t = title(['Precision@10 (Prevalence = ',sprintf('%0.3f)\n',(1-p_missing(p_miss))/10)]);
set(t,'position',get(t,'position')-[0 axis_b/50 0])
xlabel('Sample size (N)'); %ylabel('Precision at 10');
xticks(1:numel(sample_size))
set(gca,'XTickLabel',sprintf('%d\n',sample_size),'fontsize',fontsize)
legend(methods_name,'FontSize',legendfontsize,'Location','southeast','interpreter','latex');legend('boxoff')
grid on

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% filename = '/Users/yuksel/Documents/Dataset/inception_net_2048/%s_MNIST_Fashion_inception_2048.mat';
% 
% datanames = {'x_train','x_test'};
% labelnames = {'y_train','y_test'};
% %labeldata = {'images','labels'};
% datatype  = {'train','test'};
% for i = 1:numel(datanames)
%     load(sprintf(filename,datanames{i}))
%     %traindata = cat(1,data,traindata);
%     dataset.(datatype{i}).('images')= data;
%     
%     load(sprintf(filename,labelnames{i}))
%     %testdata = cat(1,data,testdata);
%     dataset.(datatype{i}).('labels')= data;
% end
% save('Fashion_MNIST_inception_2048.mat','dataset')
%%
% Fashion_Mnist= {
% '0'	'T-shirt/top'
% '1'	'Trouser'
% '2'	'Pullover'
% '3'	'Dress'
% '4'	'Coat'
% '5'	'Sandal'
% '6'	'Shirt'
% '7'	'Sneaker'
% '8'	'Bag'
% '9'	'Ankle boot'};

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

















%%