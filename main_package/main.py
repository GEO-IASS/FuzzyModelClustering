'''
Created on 18-Jun-2015
@author: Steven Spielberg P
email:stevenspielberg8@gmail.com
'''

import os
import sys

basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath,".."))
if filepath not in sys.path:
    sys.path.append(filepath)

#importing files
from DataReader.Dataset_Reader import *
from PreProcessingTechniques.Processing_dataset import *
from InputOutputModels.Models_File import *
from InputOutputModels import Fuzzy_Model_clustering as FMC
import numpy 
import pandas as pd
import math
import pylab as pl    

#----Simulated Data----
file_path1="s_input.txt" #specify input data here
file_path2="s_output.txt"#specify target(label) data here
input_data=pd.read_csv(file_path1,sep='\n',header=None,dtype="float64");
input_data=numpy.array(input_data)
output_data=pd.read_csv(file_path2,sep='\n',header=None,dtype="float64");
output_data=numpy.array(output_data)

#---- scaling data (to make it in the range 0-1)----
 
row=input_data.shape[0]
col=input_data.shape[1]
input_data_scale=numpy.zeros([row,col])
i=0
j=0
for j in range(0,col):
    maxi=max(input_data[:,j])
    mini=min(input_data[:,j])
    if maxi ==  mini:
        input_data_scale[:,j]=numpy.zeros([row])
    else:
        for i in range(0,row):
            input_data_scale[i,j]=(input_data[i,j]-mini)/(maxi-mini)




no_iter=input('Enter the number of Nth cluster: ')
cluster_datapoint=[None]*no_iter
cluster_no=[None]*no_iter
rmse_mat=[None]*no_iter
for i in range(0,no_iter):
    no_of_cluster=i+1
    input_cluster,output_cluster,model_coeff,no_cp,dim,rmse_mag=FMC.main(input_data, output_data,no_of_cluster)
    cluster_no[i]=no_of_cluster
    rmse_mat[i]=rmse_mag     #matrix containing rmse values w.r.t cluster centre     
    oc=output_cluster
    ic=input_cluster
    C=model_coeff    

## oo-list
    oo=[None]*no_cp
    for i in range(0,no_cp):
        oo[i]=numpy.array(oc['C'+str(i+1)])

## ii-list
    ii=[None]*no_cp
    for i in range(0,no_cp):
        ii[i]=numpy.array(ic['C'+str(i+1)])
    
##stacking models and number of data points
    dp=numpy.zeros([no_cp,1])
    for i in range(0,no_cp):
        dp[i]=oo[i].shape[0]
        c_dp=numpy.zeros([no_cp,dim+1])
        c_dp=numpy.hstack((model_coeff,dp))     
    cluster_datapoint[i]=dp                    #contains number of datapoints associated with each clusters
#plot rmse vs no. of cluster
pl.plot(cluster_no,rmse_mat)    

