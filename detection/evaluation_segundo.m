
figure()
 hold on;
output=load('G:\khizer\hyperspectral_datasets\E_CEM-for-Hyperspectral-Target-Detection-master\MF_segundo_pca_output.mat')
 Calc=cal_AUC(output.output(:),logical(output.groundtruth(:)),1,1)
 [PD1_MF,PF1_MF,tau1_MF]=plt_3DROC(output.output(:),logical(output.groundtruth(:)),'MF',1);

 output=load('G:\khizer\hyperspectral_datasets\E_CEM-for-Hyperspectral-Target-Detection-master\CEM_segundo_pca_output.mat')
 Calc=cal_AUC(output.output(:),logical(output.groundtruth(:)),1,1)
 [PD1_CEM,PF1_CEM,tau1_CEM]=plt_3DROC(output.output(:),logical(output.groundtruth(:)),'CEM',1);
 
  output=load('G:\khizer\hyperspectral_datasets\E_CEM-for-Hyperspectral-Target-Detection-master\ACE_segundo_pca_output.mat')
 Calc=cal_AUC(output.output(:),logical(output.groundtruth(:)),1,1)
 [PD1_ACE,PF1_ACE,tau1_ACE]=plt_3DROC(output.output(:),logical(output.groundtruth(:)),'ACE',1);
 
  output=load('G:\khizer\hyperspectral_datasets\E_CEM-for-Hyperspectral-Target-Detection-master\E-CEM_segundo_pca_output.mat')
 Calc=cal_AUC(output.output(:),logical(output.groundtruth(:)),1,1)
 [PD1_E_CEM,PF1_E_CEM,tau1_E_CEM]=plt_3DROC(output.output(:),logical(output.groundtruth(:)),'E-CEM',1);
 
   output=load('E:\Compressed\hCEM_Demo\segundo_pca_hcem.mat')
 Calc=cal_AUC(output.outputs(:),logical(output.targets(:)),1,1)
 [PD1_HCEM,PF1_HCEM,tau1_HCEM]=plt_3DROC(output.outputs(:),logical(output.targets(:)),'h-CEM',1);
 
  % output=load('G:\khizer\HTD-Net\hydice_htdnet_output.mat')
 %Calc=cal_AUC(output.output(:),logical(output.groundtruth(:)),1,1)
 %[PD1_HTD_NET,PF1_HTD_NET,tau1_HTD_NET]=plt_3DROC(output.output(:),logical(output.groundtruth(:)),'HTD-Net',1);
 
     output=load('G:\khizer\HSI-detection-main\segundo_pca_dmbdl.mat')
 Calc=cal_AUC(output.re(:),logical(output.groundtruth(:)),1,1)
 [PD1_dmbdl,PF1_dmbdl,tau1_dmbdl]=plt_3DROC(output.re(:),logical(output.groundtruth(:)),'DM-BDL',1);
 
      output=load('F:\BLTSC-master\New folder\segundo_bltsc.mat')
 Calc=cal_AUC(output.B(:),logical(output.map(:)),1,1)
 [PD1_BLTSC,PF1_BLTSC,tau1_BLTSC]=plt_3DROC(output.B(:),logical(output.map(:)),'BLTSC',1);
 
       output=load('F:\BLTSC-master\New folder\segundo_proposed4.mat')
 Calc=cal_AUC(output.B(:),logical(output.map(:)),1,1)
 [PD1_PROPOSED,PF1_PROPOSED,tau1_PROPOSED]=plt_3DROC(output.B(:),logical(output.map(:)),'PROPOSED',1);
 
  figure(1)
 semilogx(PF1_MF,PD1_MF,PF1_CEM,PD1_CEM,PF1_ACE,PD1_ACE,PF1_E_CEM,PD1_E_CEM,PF1_HCEM,PD1_HCEM,PF1_dmbdl,PD1_dmbdl,'cyan',PF1_BLTSC,PD1_BLTSC,'green',PF1_PROPOSED,PD1_PROPOSED,'blue','LineWidth',2.0)
 xlim([10^-4,10^0])
 ylabel('Probability of Detection')
 xlabel('False Alarm Rate') 
 legend('MF','CEM','ACE','E-CEM','h-CEM','DM-BDL','BLTSC','Proposed')
 %saveas(figure(1),'segundo_pd_pf.png')
 
 figure(7)
 plot(tau1_MF,PD1_MF,tau1_CEM,PD1_CEM,tau1_ACE,PD1_ACE,tau1_E_CEM,PD1_E_CEM,tau1_HCEM,PD1_HCEM,tau1_dmbdl,PD1_dmbdl,'cyan',tau1_BLTSC,PD1_BLTSC,'green',tau1_PROPOSED,PD1_PROPOSED,'blue','LineWidth',2.0)
 %xlim([10^-4,10^0])
 ylabel('Probability of Detection')
 xlabel('Threshold')
 legend('MF','CEM','ACE','E-CEM','h-CEM','DM-BDL','BLTSC','Proposed')
 %saveas(figure(2),'segundo_pd_tau.png')
 
 figure(3)
 plot(tau1_MF,PF1_MF,tau1_CEM,PF1_CEM,tau1_ACE,PF1_ACE,tau1_E_CEM,PF1_E_CEM,tau1_HCEM,PF1_HCEM,tau1_dmbdl,PF1_dmbdl,'cyan',tau1_BLTSC,PF1_BLTSC,'green',tau1_PROPOSED,PF1_PROPOSED,'blue','LineWidth',2.0)
 %xlim([10^-4,10^0])
 ylabel('False Alarm Rate')
 xlabel('Threshold')
 legend('MF','CEM','ACE','E-CEM','h-CEM','DM-BDL','BLTSC','Proposed')
 %saveas(figure(3),'segundo_pf_tau.png')