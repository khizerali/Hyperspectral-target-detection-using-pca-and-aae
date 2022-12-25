% clear
close all
%% Load the hyperspectral image and ground truth
addpath('../functions')
addpath('../data')
% load segundo_pca_20bands
% load ../coarse/segundo_pca_20bands/result_coarse
% load ../result/segundo_pca_20bands/reconstruct_result_huber_th7
load indian_pines_c16_pca1
load ../coarse/\indian_pines_pca/result_coarse
load ../result/\indian_pines_pca/reconstruct_result
 map=(groundtruth);
%map=transpose(map);
%load San
% data=transpose(X);
% data=reshape(data,[200,200,189]);

 data=double(data);
 d=double(d);
tic
[w, h, bs] = size(data);
data = hyperNormalize( data );
data_r = hyperConvert2d(data)';
reconstruct_result = hyperNormalize(reconstruct_result)*1;
%for lamda=5:5:25
%% Parameters setup
lamda=10;
maxi = 10;

%% Difference
for i = 1: w*h
    sam(i)= hyperSam(data_r(i,:), reconstruct_result(i,:));
end 
sam = reshape(sam , w, h);
SAM = hyperNormalize( sam )*1;
%% Binary the difference 
%result_coarse=abs(result_coarse);
%map=logical(groundtruth);
result_coarse=hyperNormalize(result_coarse)*1;
output  = nonlinear(result_coarse, lamda, maxi );
B = (1.0*(SAM).* output);
%B=output-0.1*SAM;
%B=(B-0.5*(sam));
%B=B/max(B(:));
toc
[FPR,TPR,thre] = myPlot3DROC( logical(map), B);
auc = -trapz(FPR,TPR);
fpr = -trapz(FPR,thre);

%figure, imagesc(B), axis image, axis off
   
 figure, imshow(logical(map))
 figure, imshow((B))
 Calc=cal_AUC(B(:),logical(map(:)),1,1)
 % plot_3DROC(B(:),logical(map(:)),'BLTSC-PCA',1);
%end
%
% similarity=[];
% count=0;
% thr=0.00001:0.01:1;
% %thr=logspace(-5,0,100);
% for th=0.00001:0.01:1;
%     yy=im2bw(B,th);
%     count=count+1;
% 
%     
%     similarity(count) = jaccard(yy,logical(map));
%     
% end
% figure()
% semilogx(FPR(:),TPR(:))
% auc_under_iou = trapz(thr,similarity);
%     figure;
%     
%     semilogx(thr,similarity)
%         
% 
%     iou_max=max(similarity(:));

% 
