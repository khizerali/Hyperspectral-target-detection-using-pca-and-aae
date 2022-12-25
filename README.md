This project provides an implementation for "Hyperspectral target detection using PCA and AAE". 
# Instructions for Proposed Algorithm:
1. Convert dataset into low dimensions using pca if not available. In order to convert dataset, use PCA_Dataset.py file.
2.Run demo.m in coarse folder to get background training samples.
3.run train.py to train AAE.
4.Run test.py to get output file (inference).
5.For the final result , run demo.m from detection folder.

The code for classical algorithms for HTD are utilized from :https://github.com/Rui-ZHAO-ipc/E_CEM-for-Hyperspectral-Target-Detection.git

