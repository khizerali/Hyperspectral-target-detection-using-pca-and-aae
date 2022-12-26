This project provides an implementation for "Hyperspectral target detection using PCA and AAE". 
# Instructions for Proposed Algorithm:
1. Convert dataset into low dimensions using pca if not available. In order to convert dataset, use PCA_Dataset.py file.
2. Run demo.m in coarse folder to get background training samples. Change the dataset path for the available dataset in this code.
3. Run train.py to train AAE. 
4. Run test.py to get output file (inference).
5. For the final result , run demo.m from detection folder.

The code for classical algorithms for HTD are utilized from :https://github.com/Rui-ZHAO-ipc/E_CEM-for-Hyperspectral-Target-Detection.git
DM-BDL code:https://github.com/FDU-ctk/HSI-detection



The datasets used in this work can be downloaded from following links:
https://pern-my.sharepoint.com/:u:/g/personal/khizer15_ist_edu_pk/EYHdTgLnUUFHokUpThF5MAUBcWUa8M3ObBGGyqhyyqu3sQ?e=11m7dQ
https://pern-my.sharepoint.com/:u:/g/personal/khizer15_ist_edu_pk/ERVbmWrBqwdFjvp5EKJyubEBYrpT_K1NRTN3WGfEIduXaw?e=Hu4G6S
https://pern-my.sharepoint.com/:u:/g/personal/khizer15_ist_edu_pk/EerQk6b6lhtEjggwjwQXWbEBqg2InTsqCzNUCEjaI-D-mA?e=ZyRjIm
https://pern-my.sharepoint.com/:u:/g/personal/khizer15_ist_edu_pk/EeUZYrRv5FpIlSvT29_fVsMBmoQ5qj6S1JOFxW3ZblW1vA?e=Vfk9nH
