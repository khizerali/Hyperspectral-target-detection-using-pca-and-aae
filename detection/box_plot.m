%% Statistical Separability Analysis(Boxplot)
% Compiled by Zephyr Hou on 2019-01-23
%% Load Results



% load('Res_CatIsland.mat')
% load('Res_TexasCoast.mat')
% load('Res_PaviaCentra.mat')
% load('Res_Cri.mat')
rows=512;
cols=217;

num_meth = 8;   % The number of all methods

%%
output=load('G:\khizer\hyperspectral_datasets\E_CEM-for-Hyperspectral-Target-Detection-master\MF_salinas_c6_output.mat')
 R1=output.output;
 
 output=load('G:\khizer\hyperspectral_datasets\E_CEM-for-Hyperspectral-Target-Detection-master\CEM_salinas_c6_output.mat')
 R2=output.output;
 
   output=load('G:\khizer\hyperspectral_datasets\E_CEM-for-Hyperspectral-Target-Detection-master\ACE_salinas_c6_output.mat')
 R3=output.output;
 
   output=load('G:\khizer\hyperspectral_datasets\E_CEM-for-Hyperspectral-Target-Detection-master\E-CEM_salinas_c6_output.mat')
 R4=output.output;
 
    output=load('E:\Compressed\hCEM_Demo\salinas_c6_hcem.mat')
 R5=output.outputs;
 
      output=load('G:\khizer\HSI-detection-main\salinas_c6_dmbdl.mat')
 R6=output.re;
 
       output=load('F:\BLTSC-master\New folder\salinas_c6_bltsc.mat');
 R7=output.B;
 
        output=load('F:\BLTSC-master\New folder\salinas_c6_proposed.mat');
 R8=output.B;
%% Normalized(0-1)
max1 = max(max(R1));min1 = min(min(R1));
R1 = (R1-min1)/(max1-min1);
max2 = max(max(R2));min2 = min(min(R2));
R2 = (R2-min2)/(max2-min2);
max3 = max(max(R3));min3 = min(min(R3));
R3 = (R3-min3)/(max3-min3);
max4 = max(max(R4));min4 = min(min(R4));
R4 = (R4-min4)/(max4-min4);
max5 = max(max(R5));min5 = min(min(R5));
R5 = (R5-min5)/(max5-min5);
max6 = max(max(R6));min6 = min(min(R6));
R6 = (R6-min6)/(max6-min6);
max7 = max(max(R7));min7 = min(min(R7));
R7 = (R7-min7)/(max7-min7);
max8 = max(max(R8));min8 = min(min(R8));
R8 = (R8-min8)/(max8-min8);


%% Reshape 2D result to vector
label_value = reshape(map,1,rows*cols);
R1_value = reshape(R1,1,rows*cols);
R2_value = reshape(R2,1,rows*cols);
R3_value = reshape(R3,1,rows*cols);
R4_value = reshape(R4,1,rows*cols);
R5_value = reshape(R5,1,rows*cols);
R6_value = reshape(R6,1,rows*cols);
R7_value = reshape(R7,1,rows*cols);
R8_value = reshape(R8,1,rows*cols);


%%
ind_tar = find(label_value == 1);
ind_bac = find(label_value == 0);
num_targ = length(ind_tar);
num_back = length(ind_bac);

targ1 = R1_value(ind_tar);
targ2 = R2_value(ind_tar);
targ3 = R3_value(ind_tar);
targ4 = R4_value(ind_tar);
targ5 = R5_value(ind_tar);
targ6 = R6_value(ind_tar);
targ7 = R7_value(ind_tar);
targ8 = R8_value(ind_tar);


back1 = R1_value(ind_bac);
back2 = R2_value(ind_bac);
back3 = R3_value(ind_bac);
back4 = R4_value(ind_bac);
back5 = R5_value(ind_bac);
back6 = R6_value(ind_bac);
back7 = R7_value(ind_bac);
back8 = R8_value(ind_bac);


X_targ = [targ1;targ2;targ3;targ4;targ5;targ6;targ7;targ8]';
X_back = [back1;back2;back3;back4;back5;back6;back7;back8]';
X = [X_targ(:);X_back(:)];
X = X(:);
g1_targ = [ones(1,num_targ); 2*ones(1, num_targ); 3*ones(1, num_targ);4*ones(1, num_targ);...
    5*ones(1, num_targ);6*ones(1, num_targ);7*ones(1, num_targ);8*ones(1, num_targ)]'; 
g1_back = [ones(1, num_back); 2*ones(1, num_back); 3*ones(1, num_back);4*ones(1, num_back);...
    5*ones(1, num_back);6*ones(1, num_back);7*ones(1, num_back);8*ones(1, num_back)]'; 
g1 = [g1_targ(:); g1_back(:)];
g1 = g1(:);
g2 = [ones(num_meth*num_targ,1);2*ones(num_meth*num_back,1)];
g2 = g2(:);
positions = [[1:num_meth],[1:num_meth]+0.3];
%%
figure(2);
% ????['plotstyle','compact']['colorgroup',g2,]['color','rk']
bh=boxplot(X, {g2,g1} ,'whisker',10000,'colorgroup',g2, 'symbol','.','outliersize',4,'widths',0.2,'positions',positions,'color','rb'); 

set(bh,'LineWidth',1.5)
ylabel('Detection test statistic range');

% grid on
% set(gca,'YLim',[0,0.5],'gridLineStyle', '-.');

% ylim([0,0.0065])  % ??y????????????

Xtick_pos = [1:num_meth]+0.15;% ??label?????
Xtick_label  ={'MF','CEM','ACE','E-CEM','h-CEM','DM-BDL','BLTSC','Proposed'};
set(gca,'XTickLabel',Xtick_label, 'XTick',Xtick_pos); %????????['fontsize',15]
% xtickangle(15)% ??????

%% 
h=findobj(gca,'Tag','Outliers');
delete(h) 
legend(findobj(gca,'Tag','Box'),'Background','Target')

%% ?????????????????whisker ?0-100%?
p_targ = prctile(X_targ,[0 100]);
p_back = prctile(X_back,[0 100]);
% p_targ = prctile(X_targ,[10 90]);
% p_back = prctile(X_back,[10 90]);
p = [];
for i = 1:num_meth
    p = [p,p_targ(:,i),p_back(:,i)];
end

% ?????????? (???????10% ? 90% ??)
q_targ = quantile(X_targ,[0.1 0.9]);  
q_back = quantile(X_back,[0.1 0.9]);  
% q_targ = quantile(X_targ,[0.09 0.81]);  
% q_back = quantile(X_back,[0.09 0.81]);  
q = [];
for i = 1:num_meth
    q = [q,q_targ(:,i),q_back(:,i)];
end

h = flipud(findobj(gca,'Tag','Box'));
for j = 1:length(h)
    q10 = q(1,j);
    q90 = q(2,j);
    set(h(j),'YData',[q10 q90 q90 q10 q10]);
end

% Replace upper end y value of whisker
h = flipud(findobj(gca,'Tag','Upper Whisker'));
for j=1:length(h);
%     ydata = get(h(j),'YData');
%     ydata(2) = p(2,j);
%     set(h(j),'YData',ydata);
    set(h(j),'YData',[q(2,j) p(2,j)]);
end

% Replace all y values of adjacent value
h = flipud(findobj(gca,'Tag','Upper Adjacent Value'));
for j=1:length(h);
%     ydata = get(h(j),'YData');
%     ydata(:) = p(2,j);
    set(h(j),'YData',[p(2,j) p(2,j)]);
end

% Replace lower end y value of whisker
h = flipud(findobj(gca,'Tag','Lower Whisker'));
for j=1:length(h);
%     ydata = get(h(j),'YData');
%     ydata(1) = p(1,j);
    set(h(j),'YData',[q(1,j) p(1,j)]);
end

% Replace all y values of adjacent value
h = flipud(findobj(gca,'Tag','Lower Adjacent Value'));
for j=1:length(h);
%     ydata = get(h(j),'YData');
%     ydata(:) = p(1,j);
    set(h(j),'YData',[p(1,j) p(1,j)]);
end