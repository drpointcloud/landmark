
clc
clear
% format short
%load('/Users/yuksel/Documents/Dataset/FashionMNIST/fashion_mnist.mat')
%load('/Users/yuksel/Documents/Dataset/CIFAR/cifar_inception_2048.mat');
%load('/Users/yuksel/Documents/Dataset/FashionMNIST/fashion_MNIST_inception_2048.mat');
%load('C:\Users\yuksel\Google Drive\Resources\Implementation\MATLAB\Landmark_clean\emnist-mnist.mat');
load('/Users/yuksel/Documents/Dataset/EMNIST/emnist-mnist.mat');

rng(0);  % set the random seed
%%
% Requires minFunc_2012
%  addpath(genpath('path_to/minFunc_2012'))
%%
sample_size = round(100*linspace(1,50,25)); % Number of samples from p_X
nperms = 1;
p_missing = 0.85;%linspace(0,1,9);% for scenario two % the digit has 100*(1-q)% of
nmonte = 1;%10*numel(p_missing);  % must be divisible by 3
classes = ['0','1','2','3','4','5','6','7','8','9'];

%%
methods = {
    %{@(K,x_idx,method) kernel_max_proj_landmark(K,x_idx,method,nperms), 'L-Bures',@(x) x};
    {@(K,x_idx,method) kernel_max_proj_landmark(K,x_idx,method,nperms), 'L-W2',@(x) x};
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
    % Wasserstein distance between probability measures on RKHS
    {@(DKw) hungarian(DKw),'Discrete W2',@(x)x};
    %A,K - a square cost matrix.%C,P - the optimal assignment.
    %T,D - the cost of the optimal assignment.
    %s.t. T = trace(A(C,:)) is min over all possible assignments.
    
    };
total_methods = methods;
Nmthds = numel(total_methods);
Npmiss = numel(p_missing);

%%
Time_spnt = nan(numel(sample_size),numel(p_missing),numel(classes),nmonte,Nmthds);

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

            time_spnt = nan(nmonte,Nmthds);          
            
            aaa=aaa+1;
            for monte_ii = 1:nmonte

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
                for method_ii = 1:Nmthds 
                
                Z = double(cat(1,X,Y)); % combine two samples
                x_idx = cat(1,ones(size(X,1),1),zeros(size(Y,1),1))==1; %bool indicator
                

                
        
                %% methods
                
                    if method_ii<3
                        
                        method_way = methods{method_ii}{1};
                        method = methods{method_ii}{2};
                        
                        tic
                        [K,sigma] = gaussian_kernel(Z,'median');
                
                        [V,~,~,~] =  method_way(K,x_idx,method);
                        %[V,divs,alpha,D1] =  method_way(K,x_idx,method);
                        time_spnt(monte_ii, method_ii) = toc;
                        %d_nperms(monte_ii, method_ii,:) = D1;
                        % if mean(D1>=div), then reject the H_0 (null hypothesis)
                        %betahat(monte_ii, method_ii) = mean(D1>=divs);
                        %div_value(monte_ii, method_ii) = divs;
                        %witness{monte_ii,method_ii} = V;
                        %alphas{monte_ii,method_ii} = alpha;
                    end
                    if method_ii ==3 && sample_ii<8
                        tic
                        [K,sigma] = gaussian_kernel(Z,'median');
                        Dkw = -2*K(x_idx, ~x_idx) + diag(K(x_idx,x_idx)) + diag(K(~x_idx,~x_idx))';
                        [~,D3]=hungarian(Dkw); %[C,T]=hungarian(A)
                        time_spnt(monte_ii, method_ii) = toc;
                    end
                    
                end  %methods
            end% nmonte
            %1-ssize,2-pmiss,3-class,4-nmonte,5-methods,6-2
            Time_spnt(sample_ii, pmiss_ii,label_ii,:,:) = time_spnt;
            
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
fontsize = 20;
linewidth = 3;
markersize = 15;
legendfontsize = 18;
%save('power_mnist_full_LY.mat')
methods_name = arrayfun(@(i) total_methods{i}{2},(1:Nmthds),'uni',0);

%% computation time as a function of sample size
%(samp_sz,pmiss,class,nmonte,methods)
sequencc = [3 1 2];
methodsname = methods_name(sequencc);
figure(22),clf
%to adjust subplot positions: do not change
n_row = 1; n_col = 2; % # of columns and rows
fig_width = 0.8; fig_height= 0.65;
pad_w = (1-fig_width)/(n_col+1); pad_h = (1-fig_height)/(n_row+1);
axis_w = 1/n_col*fig_width; axis_h = 1/n_row*fig_height;


i=1;
%to adjust subplot positions
row = floor((i-1)/n_col)+1; col = mod(i-1,n_col)+1;
axis_l = axis_w*(col-1) + pad_w*col; axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);

kk =0;
digit_ii = 1;
subplot('position', [axis_l axis_b axis_w axis_h] )
for method_ii=1:numel(total_methods)
    kk=kk+1;
    plot(Time_spnt(:,1,digit_ii,1,sequencc(method_ii)),':','linewidth',linewidth,...
        'Marker',markers{kk},'MarkerSize',markersize);hold on
end

t = title(['Computation Time (sc) (digit ',sprintf('"%d")',digit_ii-1)]);
set(t,'position',get(t,'position')-[0 0 0])
xlabel('Sample size (N)');
xticks(1:8:numel(sample_size))
set(gca,'XTickLabel',sprintf('%d\n',sample_size(1:8:end)),'fontsize',fontsize)
legend(methodsname,'FontSize',legendfontsize,'Location','southeast','Interpreter','latex');legend('boxoff')
grid on

i=2;
%to adjust subplot positions
row = floor((i-1)/n_col)+1; col = mod(i-1,n_col)+1;
axis_l = axis_w*(col-1) + pad_w*col; axis_b = axis_h*(n_row-row) + pad_h*(n_row-row+1);

kk =0;
subplot('position', [axis_l axis_b axis_w axis_h] )
for method_ii=1:numel(total_methods)
    kk=kk+1;
    plot(mean(Time_spnt(:,1,:,1,sequencc(method_ii)),3),':','linewidth',linewidth,...
        'Marker',markers{kk},'MarkerSize',markersize);hold on
end

t = title('Averaged Computation Time (sc)');
% set(t,'position',get(t,'position')-[0 axis_b/100 0])
xlabel('Sample size (N)');
xticks(1:8:numel(sample_size))
set(gca,'XTickLabel',sprintf('%d\n',sample_size(1:8:end)),'fontsize',fontsize)
legend(methodsname,'FontSize',legendfontsize,'Location','southeast','Interpreter','latex');legend('boxoff')
grid on


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
