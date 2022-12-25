x=[5,10,15,20,25];
pd_pf_hydice=[0.9736,0.9997,0.9992,0.9974,0.9956];
pd_pf_segundo=[0.9962,0.9960,0.9954,0.9958,0.9958];
pd_pf_salinas=[0.9970,0.9976,0.9976,0.9976,0.9975];
pd_pf_indian_pines=[0.9946,0.9919,0.9908,0.9906,0.9906];
pd_pf_san=[0.8839, 0.9976,0.9976,0.9958,0.9912];
figure(5)
plot(x,pd_pf_hydice,x,pd_pf_segundo,x,pd_pf_salinas,x,pd_pf_indian_pines,x,pd_pf_san,'LineWidth',1.5)
 ylabel('Area under Pd,Pf Curve')
 xlabel('lamda')
 legend('HYDICE Urban','Segundo','Salinas','Indian Pines','San Diego')
 saveas(figure(5),'lamda_pd_pf.png')
 
 pd_tau_hydice=[0.2893,0.4530,0.4642,0.4670,0.4648];
 pd_tau_segundo=[0.1005,0.2992,0.3088,0.3095,0.3096];
 pd_tau_salinas=[1.1000e-05,0.3708,0.5084,0.5265,0.5292];
 pd_tau_indian_pines=[0.2999,0.4876,0.4954,0.4959,0.4959];
 pd_tau_san=[0.1023,0.2206,0.2429,0.2498,0.2518];
 figure(6)
 plot(x,pd_tau_hydice,x,pd_tau_segundo,x,pd_tau_salinas,x,pd_tau_indian_pines,x,pd_tau_san,'LineWidth',1.5)
 ylabel('Area under Pd,tau Curve')
 xlabel('lamda')
 legend('HYDICE Urban','Segundo','Salinas','Indian Pines','San Diego')
 saveas(figure(6),'lamda_pd_tau.png')
 pf_tau_hydice=[0,2.2400e-04,0.0059,0.0233,0.0357];
 pf_tau_segundo=[5.4000e-05,0.0331,0.0451,0.0465,0.0467];
 pf_tau_salinas=[1.5000e-05,0.0022,0.0105,0.0514,0.0945];
 pf_tau_indian=[0.0016,0.0399,0.0950,0.1249,0.1368];
 pf_tau_san=[0,1.7200e-04,6.6800e-04,0.0036,0.0112];
 figure(7)
  plot(x,pf_tau_hydice,x,pf_tau_segundo,x,pf_tau_salinas,x,pf_tau_indian,x,pf_tau_san,'LineWidth',1.5)
 ylabel('Area under pf,tau Curve')
 xlabel('lamda')
 legend('HYDICE Urban','Segundo','Salinas','Indian Pines','San Diego')
 saveas(figure(7),'lamda_pf_tau.png')
 
 
 %%%
