# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 22:14:54 2021

@author: Khizer
"""

from scipy.io import loadmat,savemat
import numpy as np
import cv2
#load dataset available in mat format
data_path=r'C:\Users\Khizer\Downloads\Compressed\OneDrive_2021-12-25\Proposed algorithmdata\salinas_c6_1.mat'
input_dataset=loadmat(data_path)
data=input_dataset['data']
d=input_dataset['d'] # target spectrum 
map=input_dataset['map'] #groundtruth

#%%
from sklearn.decomposition import PCA
n_components=20
pca = PCA(n_components)
pca.fit(data.reshape(data.shape[0]*data.shape[1],data.shape[2]))
reduced_pca_data=pca.transform(data.reshape(data.shape[0]*data.shape[1],data.shape[2]))
target_spectrum_pca=pca.transform(d.reshape(d.shape[1],d.shape[0]))

reduced_pca_data=reduced_pca_data.reshape(data.shape[0],data.shape[1],n_components)
mdict={'data':reduced_pca_data,'d':target_spectrum_pca.reshape(n_components,1),'map':map}

savemat('salinas_pca_20_components_Dataset',mdict)