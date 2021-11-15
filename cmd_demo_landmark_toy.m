clear
clc

%%
% Generate data in two distribution whose support looks like Venn diagrams
savedir =   '/Users/yuksel/Desktop/';
rng(10);
n = 500;
d = 2;
p = 500;
nperms = 1;
%sigma = 0.2;
% weights = randn(d,p)/sigma;
weights = @(sigma)randn(d,p)/sigma;
centers = rand(1,p)*2*pi;
% rand_Fourier = @(X) cos(X*weights+centers)*sqrt(2/p);
rand_Fourier = @(X,sigma) cos(X*weights(sigma)+centers)*sqrt(2/p);

%%
methods_landmark = {
    {@(K,x_idx,method) kernel_max_proj_landmark(K,x_idx,method,nperms),'L-W2-max','L-W2',@(x) x};
    %{@(K,x_idx,method) kernel_max_proj_landmark(K,x_idx,method,nperms),'L-Bures','L-Bures',@(x) x};
    };

methods1 = {
    {@(K,x_idx,method) kernel_max_slice(K,x_idx,method), 'MMD','MMD',@(x) x};
    {@(K,x_idx,method) kernel_max_slice(K,x_idx,method), 'W2','W2',@(x) x};
    %{@(K,x_idx,method) kernel_max_slice(K,x_idx,method), 'Bures','Bures',@(x) x};
    };

methods_landmark_rfb = {
    {@(K,x_idx,method) kernel_max_proj_landmark(K,x_idx,method,nperms),'L-W2-max','L-W2-RFB',@(x) x};
    %{@(K,x_idx,method) kernel_max_proj_landmark(K,x_idx,method,nperms),'L-Bures','L-Bures-RFB',@(x) x};
    };

methods1_rfb = {
    {@(K,x_idx,method) kernel_max_slice(K,x_idx,method), 'MMD','MMD-RFB',@(x) x};
    {@(K,x_idx,method) kernel_max_slice(K,x_idx,method), 'W2','W2-RFB',@(x) x};
    %{@(K,x_idx,method) kernel_max_slice(K,x_idx,method), 'Bures','Bures-RFB',@(x) x};
    };

methods = cat(1,methods_landmark,methods1,methods_landmark_rfb,methods1_rfb);
methods_name = arrayfun(@(i) methods{i}{3},(1:numel(methods)),'uni',0);

%% Create data
% Create uniform with circular support using rejection sampling
Z = 2*(rand(4*5*n,2)-0.5);
idx = sum(Z.^2,2)<=1;
Z = Z(idx,:);
% Circle offsets
offsets = 1.5*[1 0 ; 0 0; cos(pi/3) sin(pi/3)];
X = Z(1:n,:) + offsets(randi(3,n,1), :); % X ~ \mu is uniformly across the support
%Y ~ \nu has mass [0, 1/3, 2/3] in each circle

m = round(n*1); %n*1 is when m=n
Y = Z(m+1:2*m,:) + offsets(2+(rand(m,1)>1/3), :);


%%
Time_spnt = nan(numel(methods));
Witness = cell(size(Time_spnt));
Alphas = Witness;
%%

% Pool the samples and make the kernel matrix
x_idx = cat(1,ones(size(X,1),1),zeros(size(Y,1),1))==1; %indicator for X
Z = cat(1,X,Y);
[K,sigma]= gaussian_kernel(Z,'median');

RFB_X = rand_Fourier(X,sigma);
RFB_Y = rand_Fourier(Y,sigma);
RFB_Z = double(cat(1,RFB_X,RFB_Y)); % combine two samples
RFB_K = RFB_Z*RFB_Z';

% Define the kernel for test points
kappa = @(x,y,sigma) exp(-( sum(x.^2,2) + sum(y.^2,2).' -2*x*y.')/2/sigma^2);
% Make a grid of test points for contours
[X_test,Y_test]=meshgrid(linspace(min(Z(:,1)),max(Z(:,1)),40),linspace(min(Z(:,2)),max(Z(:,2)),40));


%% Landmark Method(s)
offset = 0;
for method_ii = 1:numel(methods_landmark)
    method_way = methods_landmark{method_ii}{1};
    method = methods_landmark{method_ii}{2};
    [V,~,alphas,~] =  method_way(K,x_idx,method);
    Witness{method_ii+offset} = V;
    Alphas{method_ii+offset} = alphas;
    W = Witness{method_ii+offset};
    
    omega_X_ge_Y = W;%W(W<0);%W(:,1);
    omega_X_le_Y = W;%W(W>0);%W(:,2);
    % Get coefficients for defining witness functions
    alpha1 = alphas;
    alpha2 = alphas;
    
    % Sort the witness function evaluations by their magnitude
    % Points from X ~ \mu
    [~,pi_ge] = sort(omega_X_ge_Y(x_idx).^2,'descend');
    [~,pi_le] = sort(omega_X_le_Y(x_idx).^2,'descend');
    % Points from Y ~ \nu
    [~,sigma_ge] = sort(omega_X_ge_Y(~x_idx).^2,'descend');
    [~,sigma_le] = sort(omega_X_le_Y(~x_idx).^2,'descend');
    
    % Get magnitude of witness function evaluations on test points
    omega_X_ge_Y_test = 0*X_test;
    omega_X_ge_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha1).^2;
    omega_X_le_Y_test = 0*X_test;
    omega_X_le_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha2).^2;
    
    k = 10; % Number of witness points top-k
    s = 2; %two figures (bures is two sided, so 4 figures)
    
    h = figure(10+method_ii);clf;set(h,'WindowStyle','docked') %dock the figure
    plotfigures(X,Y,s,k,X_test,Y_test,omega_X_ge_Y_test,omega_X_le_Y_test,sigma_ge,sigma_le,pi_ge,pi_le,methods_name{method_ii+offset})
    printfigure(savedir,methods_name{method_ii+offset})
end

%% Baseline Method(s)
offset = offset+numel(methods_landmark);
for method_ii = 1:numel(methods1)
    method_way = methods1{method_ii}{1};
    method = methods1{method_ii}{2};
    [V,~,alphas] =  method_way(K,x_idx,method);
    Witness{method_ii+offset} = V;
    Alphas{method_ii+offset} = alphas;
    W = Witness{method_ii+offset};

    switch method_ii
        
        %% MMD
        case 1
            
            W1 = W; W1(W1<0)=0; %positive
            W2 = W; W2(W2>0)=0; %negative
            
            omega_X_ge_Y = W1;%W(W<0);%W(:,1);
            omega_X_le_Y = W2;%W(W>0);%W(:,2);
            % Get coefficients for defining witness functions
            alpha1 = alphas;
            alpha2 = alphas;
            
            % Sort the witness function evaluations by their magnitude
            % Points from X ~ \mu
            [~,pi_ge] = sort(omega_X_ge_Y(x_idx).^2,'descend');
            [~,pi_le] = sort(omega_X_le_Y(x_idx).^2,'descend');
            % Points from Y ~ \nu
            [~,sigma_ge] = sort(omega_X_ge_Y(~x_idx).^2,'descend');
            [~,sigma_le] = sort(omega_X_le_Y(~x_idx).^2,'descend');
            
            % Get magnitude of witness function evaluations on test points
            omega_X_ge_Y_test = 0*X_test;
            omega_X_ge_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha1);
            omega_X_le_Y_test = 0*X_test;
            omega_X_le_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha2);
            
            k = 10; % Number of witness points top-k
            s = 4; %two figures (bures is two sided, so 4 figures)
            
            h = figure(20+method_ii);clf;set(h,'WindowStyle','docked') %dock the figure
            plotfigures(X,Y,s,k,X_test,Y_test,omega_X_ge_Y_test,omega_X_le_Y_test,sigma_ge,sigma_le,pi_ge,pi_le,methods_name{method_ii+offset})
            printfigure(savedir,methods_name{method_ii+offset})
            
        %% W2
        otherwise
            
            omega_X_ge_Y = W;%W(W<0);%W(:,1);
            omega_X_le_Y = W;%W(W>0);%W(:,2);
            % Get coefficients for defining witness functions
            alpha1 = alphas;
            alpha2 = alphas;
            
            % Sort the witness function evaluations by their magnitude
            % Points from X ~ \mu
            [~,pi_ge] = sort(omega_X_ge_Y(x_idx).^2,'descend');
            [~,pi_le] = sort(omega_X_le_Y(x_idx).^2,'descend');
            % Points from Y ~ \nu
            [~,sigma_ge] = sort(omega_X_ge_Y(~x_idx).^2,'descend');
            [~,sigma_le] = sort(omega_X_le_Y(~x_idx).^2,'descend');
            % Get magnitude of witness function evaluations on test points
            omega_X_ge_Y_test = 0*X_test;
            omega_X_ge_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha1).^2;
            omega_X_le_Y_test = 0*X_test;
            omega_X_le_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha2).^2;
            
            
            k = 10; % Number of witness points top-k
            s = 2; %two figures (bures is two sided, so 4 figures)
            
            h = figure(20+method_ii);clf;set(h,'WindowStyle','docked') %dock the figure
            plotfigures(X,Y,s,k,X_test,Y_test,omega_X_ge_Y_test,omega_X_le_Y_test,sigma_ge,sigma_le,pi_ge,pi_le,methods_name{method_ii+offset})
            printfigure(savedir,methods_name{method_ii+offset})
    end
end


%% Landmark RFB Method(s)
offset = offset+numel(methods1);
for method_ii = 1:numel(methods_landmark_rfb)
    method_way = methods_landmark_rfb{method_ii}{1};
    method = methods_landmark_rfb{method_ii}{2};
    [V,~,alpha,~] =  method_way(RFB_K,x_idx,method);
    Witness{method_ii+offset} = V;
    Alphas{method_ii+offset} = alpha;
    W = Witness{method_ii+offset};
    
    omega_X_ge_Y = W;%W(W<0);%W(:,1);
    omega_X_le_Y = W;%W(W>0);%W(:,2);
    
    % Get coefficients for defining witness functions
    alpha1 = alphas;
    alpha2 = alphas;
    
    % Sort the witness function evaluations by their magnitude
    % Points from X ~ \mu
    [~,pi_ge] = sort(omega_X_ge_Y(x_idx).^2,'descend');
    [~,pi_le] = sort(omega_X_le_Y(x_idx).^2,'descend');
    % Points from Y ~ \nu
    [~,sigma_ge] = sort(omega_X_ge_Y(~x_idx).^2,'descend');
    [~,sigma_le] = sort(omega_X_le_Y(~x_idx).^2,'descend');
    
    % Get magnitude of witness function evaluations on test points
    omega_X_ge_Y_test = 0*X_test;
    omega_X_ge_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha1).^2;
    omega_X_le_Y_test = 0*X_test;
    omega_X_le_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha2).^2;
    
    k = 10; % Number of witness points top-k
    s = 2; %two figures (bures is two sided, so 4 figures)
    
    h = figure(30+method_ii);clf;set(h,'WindowStyle','docked') %dock the figure
    plotfigures(X,Y,s,k,X_test,Y_test,omega_X_ge_Y_test,omega_X_le_Y_test,sigma_ge,sigma_le,pi_ge,pi_le,methods_name{method_ii+offset})
    printfigure(savedir,methods_name{method_ii+offset})
end


%% Baseline RFB Method(s)
offset = offset+numel(methods_landmark_rfb);
for method_ii = 1:numel(methods1_rfb)
    method_way = methods1_rfb{method_ii}{1};
    method = methods1_rfb{method_ii}{2};
    [V,~,alphas] =  method_way(RFB_K,x_idx,method);
    Witness{method_ii+offset} = V;
    Alphas{method_ii+offset} = alphas;
    W = Witness{method_ii+offset};

    switch method_ii
        
        %% MMD
        case 1
            W1 = W; W1(W1<0)=0; %positive
            W2 = W; W2(W2>0)=0; %negative
            
            omega_X_ge_Y = W1;%W(W<0);%W(:,1);
            omega_X_le_Y = W2;%W(W>0);%W(:,2);
            % Get coefficients for defining witness functions
            alpha1 = alphas;
            alpha2 = alphas;
            
            % Sort the witness function evaluations by their magnitude
            % Points from X ~ \mu
            [~,pi_ge] = sort(omega_X_ge_Y(x_idx).^2,'descend');
            [~,pi_le] = sort(omega_X_le_Y(x_idx).^2,'descend');
            % Points from Y ~ \nu
            [~,sigma_ge] = sort(omega_X_ge_Y(~x_idx).^2,'descend');
            [~,sigma_le] = sort(omega_X_le_Y(~x_idx).^2,'descend');
            
            % Get magnitude of witness function evaluations on test points
            omega_X_ge_Y_test = 0*X_test;
            omega_X_ge_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha1);
            omega_X_le_Y_test = 0*X_test;
            omega_X_le_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha2);
            
            k = 10; % Number of witness points top-k
            s = 4; %two figures (bures is two sided, so 4 figures)
            
            h = figure(40+method_ii);clf;set(h,'WindowStyle','docked') %dock the figure
            plotfigures(X,Y,s,k,X_test,Y_test,omega_X_ge_Y_test,omega_X_le_Y_test,sigma_ge,sigma_le,pi_ge,pi_le,methods_name{method_ii+offset})
            printfigure(savedir,methods_name{method_ii+offset})
            
            %% W2
        otherwise
            
            omega_X_ge_Y = W;%W(W<0);%W(:,1);
            omega_X_le_Y = W;%W(W>0);%W(:,2);
            % Get coefficients for defining witness functions
            alpha1 = alphas;
            alpha2 = alphas;
            
            % Sort the witness function evaluations by their magnitude
            % Points from X ~ \mu
            [~,pi_ge] = sort(omega_X_ge_Y(x_idx).^2,'descend');
            [~,pi_le] = sort(omega_X_le_Y(x_idx).^2,'descend');
            % Points from Y ~ \nu
            [~,sigma_ge] = sort(omega_X_ge_Y(~x_idx).^2,'descend');
            [~,sigma_le] = sort(omega_X_le_Y(~x_idx).^2,'descend');
            % Get magnitude of witness function evaluations on test points
            omega_X_ge_Y_test = 0*X_test;
            omega_X_ge_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha1).^2;
            omega_X_le_Y_test = 0*X_test;
            omega_X_le_Y_test(:) = (kappa([X_test(:),Y_test(:)],Z,sigma)*alpha2).^2;
            
            k = 10; % Number of witness points top-k
            s = 2; %two figures (bures is two sided, so 4 figures)
            
            h = figure(40+method_ii);clf;set(h,'WindowStyle','docked') %dock the figure
            plotfigures(X,Y,s,k,X_test,Y_test,omega_X_ge_Y_test,omega_X_le_Y_test,sigma_ge,sigma_le,pi_ge,pi_le,methods_name{method_ii+offset})
            printfigure(savedir,methods_name{method_ii+offset})
    end
end



%%  PLOT FIGURES
function plotfigures(X,Y,s,k,X_test,Y_test,omega_X_ge_Y_test,omega_X_le_Y_test,sigma_ge,sigma_le,pi_ge,pi_le,current_method)
% Color-blind safe two color scheme
%colors=[31,120,180;178,223,138]/255;

marker_size_x = 6;
marker_size_y = 6;
marker_size_c = max(marker_size_x,marker_size_y) + 4;

for ii = 1:s
    subplot(s/2,2,ii)
    plot(X(:,1),X(:,2),'oc','markersize',marker_size_x);
    hold all;
    plot(Y(:,1),Y(:,2),'+r','markersize',marker_size_y);
    set(gca,'fontsize',18);axis tight; axis equal; axis off
    %set(gca,'visible','off')
    
    switch ii
        case 1
            contour(X_test,Y_test,omega_X_ge_Y_test)
            plot(X(pi_ge(1:k),1),X(pi_ge(1:k),2),'ok','markersize',marker_size_c)
            witness_points = '$x_{\grave{\pi}(1)},\ldots,x_{\grave\pi(10)}$';
            h=legend('$X\sim \mu $','$Y\sim\nu$','$\omega^2_{\mu>\nu}$','interpreter','latex','location','southwest');
            title(h,sprintf('%s',current_method))
            set(h,'position',[ 0.4545    0.7996    0.1509    0.1340])
        case 2
            contour(X_test,Y_test,omega_X_ge_Y_test)
            plot(Y(sigma_ge(1:k),1),Y(sigma_ge(1:k),2),'ok','markersize',marker_size_c)
            witness_points = '$y_{\grave{\sigma}(1)},\ldots,y_{\grave\sigma(10)}$';
        case 3
            contour(X_test,Y_test,omega_X_le_Y_test)
            plot(Y(sigma_le(1:k),1),Y(sigma_le(1:k),2),'ok','markersize',marker_size_c)
            witness_points = '$y_{\acute{\sigma}(1)},\ldots,x_{\acute\sigma(10)}$';
            h=legend('$X\sim \mu $','$Y\sim\nu$','$\omega^2_{\mu<\nu}$','interpreter','latex','location','southwest');
            set(h,'position',[  0.4409    0.3142    0.1509    0.1340])
        case 4
            contour(X_test,Y_test,omega_X_le_Y_test)
            plot(X(pi_le(1:k),1),X(pi_le(1:k),2),'ok','markersize',marker_size_c)
            witness_points = '$x_{\acute{\pi}(1)},\ldots,x_{\acute\pi(10)}$';
    end
    text(-1.1,-1.2,witness_points,'fontsize',20,'interpreter','latex')
end
colormap(flipud(gray))

n = size(X,1);
m = size(Y,1);
dim = get(h,'Position') -  [0.1 h.Position(2) 0 0];
if m~=n
    str = ['m\neqn  ',sprintf('N_X=%d, N_Y=%d',n,m)];
    annotation('textbox',dim,'String',str,'FitBoxToText','on','FontSize',15);
else
    str = ['m=n  ',sprintf('N_X=%d, N_Y=%d',n,m)];
    annotation('textbox',dim,'String',str,'FitBoxToText','on','FontSize',15);
end

end

%% PRINT FIGURES
function printfigure(savedir,current_method)
filename = sprintf('%s.pdf',current_method);
fullfilename = fullfile(savedir,filename);
print(gcf,fullfilename,'-dpdf') % then print it
end

%%

%     [V,~,alpha,~] = L_MSKW(K,x_idx,nperms);
%     [W,~,alphas] = kernel_max_slice(K,x_idx,'mmd');
%     [V,~,alpha,~] = L_MSKW(RFB_K,x_idx,nperms);
%     [W,~,alphas] = kernel_max_slice(RFB_K,x_idx,'mmd');
%     [W,~,alphas] = kernel_max_slice(K,x_idx,'w2');
%     [W,~,alphas] = kernel_max_slice(RFB_K,x_idx,'w2');
%     [W,~,alphas] = kernel_max_slice(K,x_idx,'bures');
%     [W,~,alphas] = kernel_max_slice(RFB_K,x_idx,'bures');
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
