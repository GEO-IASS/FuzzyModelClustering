'''
Created on 18-Jun-2015
by Steven Spielberg P
'''



###################################
#######Importing Libraries#########
###################################

import math
import numpy
import random
import copy
import pprint
import fuzzy_c

###################################
########Defining functions#########
###################################

#function to read input from an excel file#

def load_into_file(path,header=True):    
    file=open(path,'r')
    lines=file.readlines()
    if header:        
        header_content=lines[0]
        del lines[0]
    final_list=[]
    for line in lines:
        final_list.append([float(i) for i in line.split('\t')])
    file.close()

    return final_list,header_content

###################################

#Function to compute the coefficent matrix if the input and output data is given#
#To find the model for given data set#

def coeff_matric(input_data_array,output_data_array):
    #.I takes the inverse of a matrix. Note: the matrix must be invertible
    #below is the std def #
    
    
    model_coeff_matrix=numpy.dot(numpy.dot(output_data_array.T,input_data_array),numpy.linalg.pinv(numpy.dot(input_data_array.T,input_data_array)))
    return model_coeff_matrix

###################################

#function to compute two norm between any two vectors of same dimension#

def norm2(vector):
    #input of this function is a vector#
    #output of this function will be a scalar, which in this case will be norm of the input vector#
    sum=0
    
    for i in range(len(vector)):
        square_of_element=(vector[i])**2
        sum=sum+square_of_element

    norm_2=math.sqrt(sum)
    return norm_2

###################################

#function to compute the predicition error matrix#

def pred_error(no_mod,no_dp,output_data_pt,input_data_pt,model_C):
    #input of this funtion is number of data points and cluster centers and input and output array#
    #output of this function will be a prediction error matrix#
    
    pred_error_matrix=numpy.zeros((no_mod,no_dp))
    
    
    for i in range(no_mod):
        for j in range(no_dp):
            
            pred_error_matrix[i][j]=numpy.linalg.norm((numpy.array([output_data_pt[j]]).T)-numpy.dot(model_C[i],numpy.array([input_data_pt[j]]).T))

    return pred_error_matrix

######################################

#function to compute membeship values#

def memb(no_mod,no_dp,pred_error_matrix,q):
    #input of this function - number of cluster centers and number of data points, prediction error matrix and fuzzi#
    #output of this function will be a matrix containing all the membership values of w.r.t. all data points#
    initial_memb=numpy.zeros((no_mod,no_dp))
    
    for i in range(no_mod):
        for j in range(no_dp):
            sum=0
            for k in range(no_mod):
                ratio=((pred_error_matrix[i][j])/(pred_error_matrix[k][j]))**(2/(q-1))
                sum=sum+ratio

            inv_denom=sum 
            initial_memb[i][j]=1/inv_denom

    return initial_memb

########################################

#function to update the coefficient matrix#

def update_model(no_mod,no_dp,old_model_C,model_C,output_data_array,input_data_array,initial_memb,q): 
    if output_data_array.shape == (len(output_data_array),):
        dim_output_data_array = 1
    else:
        dim_output_data_array = output_data_array.shape[1]     
    dim_input_data_array = len(input_data_array[0])
    for i in range(no_mod):
        old_model_C[i]=copy.deepcopy(model_C[i])
        sum1=numpy.zeros((dim_output_data_array,dim_input_data_array)) 
        sum2=numpy.zeros((dim_input_data_array,dim_input_data_array))

            
        for j in range(no_dp):
            sum1=sum1+(((initial_memb[i][j])**q)*numpy.dot(numpy.array([output_data_array[j]]).T,numpy.array([input_data_array[j]])))
            sum2=sum2+(((initial_memb[i][j])**q)*numpy.dot(numpy.array([input_data_array[j]]).T,numpy.array([input_data_array[j]])))
            
           
    
        model_C[i]=numpy.dot(sum1,numpy.linalg.pinv(sum2))
    return old_model_C,model_C


def write_to_file(path,list,header,append_flag=0):    
    if not append_flag:                
        file=open(path,'w')
    else:        
        file=open(path,'a')
    file.write('\n')
    file.write(header)
    #file.write('\n')
    for row in list:
            count=0
            for col in row:
                    count+=1
                    if count<len(row):
                            file.write(str(col)+'\t')
                    if count==len(row):
                            file.write(str(col)+'\n')
    file.close()
    
   
#function to find the rmse value of final models#

def rmse(no_dp,initial_memb,pred_error_matrix):
    pred_error_rms=numpy.zeros((1,no_dp))
    for i in range(no_dp):
        pred_error_rms[:,i]=pred_error_matrix[:,i][initial_memb[:,i].tolist().index(max(initial_memb[:,i]))]

    

    mse=0

    for i in range(no_dp):
        mse=mse+((pred_error_rms[:,i])**2)

    rmse_model=numpy.sqrt(mse/no_dp)

    return rmse_model
    
#function to get data clusters based on membership matrix

def get_data_clustersfmc(in_data,output_data,mu,reject_col_present):
    num_clusters=mu.shape[0]
    num_data_points=mu.shape[1]
    clusters_list=[]
    for i in range(num_clusters):
        clusters_list.append('C'+str(i+1))
    data_clusters={}
    output_data_clusters={}
    for col in range(num_data_points):
        mu_list=mu[:,col].tolist()
        index=int(mu_list.index(max(mu_list)))+1
        key='C'+str(index)
        if key not in data_clusters:
            data_clusters[key]=[]
            output_data_clusters[key]=[]
        if reject_col_present:
            data_clusters[key].append(in_data[col][:-1])#excluding the rejection column
            output_data_clusters[key].append(output_data[col])
        else:
            data_clusters[key].append(in_data[col])
            output_data_clusters[key].append(output_data[col])
    for key in data_clusters:
        data_clusters[key]=numpy.array(data_clusters[key])
        output_data_clusters[key]=numpy.array(output_data_clusters[key])		

    return data_clusters,output_data_clusters   

    
        


#Use above defined function to take in input_data and output_data#
def main(input_data_pt,output_data_pt,no_of_cluster):
    input_data_array=numpy.array(input_data_pt)
    
    'adding ones in the last column of input_data_array'
    ones_array = numpy.ones((input_data_array.shape[0],1))
    input_data_array = numpy.hstack((input_data_array,ones_array))
    
    output_data_array=numpy.array(output_data_pt)
    dim_input_data_array=input_data_array.shape[1] 
    dim_output_data_array=1
    

    
    #to fix the number of data points#
    
    if(input_data_array.shape[0]==output_data_array.shape[0]):
        no_dp=input_data_array.shape[0]
    elif(input_data_array.shape[0]>output_data_array.shape[0]):
        no_dp=output_data_array.shape[0]
        print('Given data set has more input data than output data ')
    else:
        no_dp=input_data_array.shape[0]
        print('Given data set has more output data than input data ')
        
    ##################################
    
    
    no_cp=no_of_cluster
    ##################################
    
    #to take in input the fuzzifier#    
    
    
    q_fuzzy_initial=2.5  #specify the fuzzifier value here
    ##################################
    
    #take in input the tolerance value for fuzzy clustering
    
    
    tol_fuzzy_initial=0.02 #specify the tolerance value for initial fuzzy clustering here
    ##################################
    
    # initialization of models using fuzzy c-mean clustering#
    
    mu_final_fuzzy_c,clstr_centers,input_clusters_dict,output_clusters_dict,header_cont=fuzzy_c.fcm_clusters_answers(input_data_array,no_cp,q_fuzzy_initial,tol_fuzzy_initial,10,0,0,output_data_pt,print_output=0,scaling_data_list=[],scale_down_input=0,scale_up_output=0,reject_col_present=0)
    
    ##################################
    
    #to check that we are getting correct output i.e. dictionary having  input and output data points, after distance based clustering
    
    for key in input_clusters_dict:
        output_data_test=numpy.array(output_clusters_dict[key])
        trans_output_data_test=output_data_test.T
        input_data_test=numpy.array(input_clusters_dict[key])
        trans_input_data_test=input_data_test.T
    
    ###################################
    
    #defining a nested list, which will have a matrix as an entry#
    
    model_C=[]
    old_model_C=[]
    for i in range(no_cp):
        model_C.append([])
        old_model_C.append([])
    
    
    for i in range(no_cp):
        model_C[i]=numpy.zeros((dim_output_data_array,dim_input_data_array))
        old_model_C[i]=numpy.zeros((dim_output_data_array,dim_input_data_array))
        key='C'+str(i+1)
        input_data=input_clusters_dict[key]
        output_data=output_clusters_dict[key]
        input_data_array_fuzzy=numpy.array(input_data)
        output_data_array_fuzzy=numpy.array(output_data)
        model_C[i]=coeff_matric(input_data_array_fuzzy,output_data_array_fuzzy)   ##line 47
    
    
   
  
    
    
    
    #no_mod is number of models/coefficient matrix returned in the end#
    
    no_mod=no_cp
    
    
    
    q=q_fuzzy_initial
    
    
    
    #Use above defined function to find the prediction error matrix#
    
    pred_error_matrix=pred_error(no_mod,no_dp,output_data_array,input_data_array,model_C)   
    
    
    #Use prediction error to find member ship values#
    
    initial_memb=memb(no_mod,no_dp,pred_error_matrix,q)                                    
    print initial_memb.shape    
    'array 3,160'
    
    
    
    #give in the expected tolerance value#
    
    tol=tol_fuzzy_initial 
    
    
    
    
    zz=0
    while 1:                              
        zz=zz+1
        print "Model no: %d Iter No: %d" % (no_cp,zz)
        
        pred_error_FMC=pred_error(no_mod,no_dp,output_data_array,input_data_array,model_C)
        
        
        flag=1
        norm=numpy.zeros((no_mod))
        
        for i in range(no_mod):
            difference=old_model_C[i][0]-model_C[i]
            norm[i]=numpy.linalg.norm(difference,2)
        
        model_norm=max(norm)
        
        
        if((model_norm)>tol):
            old_model_C,model_C=update_model(no_mod,no_dp,old_model_C,model_C,output_data_array,input_data_array,initial_memb,q) 
            #update model
            pred_error_matrix_new=pred_error(no_mod,no_dp,output_data_array,input_data_array,model_C)
            initial_memb=memb(no_mod,no_dp,pred_error_matrix_new,q)
            flag=0
            
             
        
        if(flag==1):
            print "Shape of Membership Matrix"
            print initial_memb.shape
            print "number of iteration"
            print zz
            break        

            
    #########################################
    data_cluster,output_cluster=get_data_clustersfmc(input_data_array,output_data_array,initial_memb,0)    
    #print data_cluster
    
  
    
    
    print "models are :"
    for model in model_C:        
        print model.tolist()
        
    model_CC=numpy.zeros((no_mod,dim_input_data_array))
    for i in range(0,no_mod):
        model_CC[i]=model_C[i]  
    ## Model Coeff FCM ##



## RMSE ##   
      
    pred_error_test=pred_error(no_mod,no_dp,output_data_array,input_data_array,model_C)
    
    memb_test=memb(no_mod,no_dp,pred_error_test,q)
    test_in_clust,test_out_clust=get_data_clustersfmc(input_data_array,output_data_array,memb_test,0)
    
    error=[None]*no_mod    #error between y and CX
    for i in range(0,no_mod):
        
    
        error[i]=test_out_clust[('C'+str(i+1))]-numpy.dot(model_C[i],test_in_clust[('C'+str(i+1))].T).T
        
        
    error_a=[None] *no_mod
    for i in range(0,no_mod):
        error_a[i]=numpy.array(error[i])
        
    
    rmse=numpy.zeros((no_mod,1))    
    for i in range(0,no_mod):
        
        b=int(error[i].shape[1])
        aa=numpy.square(error_a[i][0])
        aa=sum(aa)
        rmse[i]=math.sqrt(aa/b)
            
    rmse_mag=numpy.linalg.norm(rmse)
      
   
    

      
    return data_cluster,output_cluster,model_CC,no_cp,dim_input_data_array,rmse_mag
    











if __name__ == "__main__":
    pass

    