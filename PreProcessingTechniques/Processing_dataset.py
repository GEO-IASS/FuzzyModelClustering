import numpy as np
# to set precision for printing results
np.set_printoptions(precision=4, suppress=True)
def transform_output_data_to_labels(output_data,no_classes,):
        no_data=int(len(output_data)/no_classes)              
        sort_output=np.sort(output_data, axis=None) 
        Bins=map(lambda x:sort_output[x*no_data],range(int(no_classes)))
        Bins.append(max(output_data))
        hist,edges = np.histogram(output_data,bins =Bins)
        #print 'The no. datas in each class:',hist
        #print 'The class range are as:',edges
        #difference of edges
        edges_interval=map(lambda x:edges[x+1]-edges[x] ,range(len(edges)-1))
        #Make the data categorical 
        edges=edges.tolist()
        import bisect
        from numpy import array
        categorical_value=(map(lambda x: (edges.index(edges[bisect.bisect_left(edges,x)])-1),output_data))
        changed_index=categorical_value.index(-1)
        categorical_value=array(categorical_value)
        categorical_value[changed_index]=0 
        return (categorical_value,edges_interval)

#Function to remove outliers in dataset
def outlier_removal(input_output_data):
    # Cleaning process for Datsset  #
    #  1.Outliers removal   #
    '''
    If data is normal then outliers removal is very effective 
    if it is skewed then we need to convert it into transforming thing needed
    '''
    dataset=input_output_data
    '''
    removal method: outside which range of mean-3*std.dev and mean + 3*std.dev
    '''
    Mean_dataset=np.mean(dataset, axis=0) 
    Std_dataset=np.std(dataset, axis=0)

    #Range of outliers value
    Range_dataset=np.c_[Mean_dataset-3*Std_dataset,Mean_dataset+3*Std_dataset]
    dataset_list=dataset.T.tolist()

    #Outliers in each features
    dataset_outliers=np.array(map(lambda y: filter(lambda x: x<Range_dataset[y,:][0] or  x>Range_dataset[y,:][1],dataset[:,y]),range(len(dataset.T)))).T
    print 'Outliers in Dataset row-wise:'
    dataset_outlier=dataset_outliers.reshape(len(dataset_outliers),1)
    outlier_count=0
    for x in range(len(dataset_outlier)):
        temp=sum(dataset_outlier[x],[])
        outlier_count=len(temp)+outlier_count
        if temp==[]:
            print 'Outliers in',x+1,'row of dataset:', 'None'
        else:  
            print 'Outliers in',x+1,'row of dataset:',", ".join(["%.2f" % f for f in temp])

    

    #index counting for outliers
    index_num=np.zeros((outlier_count))
    index_num=index_num.tolist()
    m=0
    for x in range(len(dataset.T)):
        temp=dataset_outliers[x]
        temp1=dataset_list[x]
        for y in range(len(temp)):                           
            indices = [i for i, val in enumerate( temp1) if val == temp[y]]
            index_num[m]=indices
            m=m+1    
    #Converting a 2D list to 1D list
    index_num1D=sum(index_num,[])
    index_num1D
    #Use set to ideintical values
    indices=set(index_num1D)
    #Again converty into the list
    indices = list(indices)

    #Removing outliers from Dataset
    cl_dataset = np.delete(dataset, (indices), axis=0)
    return cl_dataset

#  2.Multicollinearity removal    #
# PCA and Kpca helps in this
def perform_pca(input_data):
    
    '''
    PCA works best when the data is numeric and distributed either Normally or Uniformly
    rather than having multiple modes,  clumps,  large outliers,  or skewed distributions.
    Moreover,  PCA works best when the data has primarily linear relationships betweeninputs and with the target variable
    Applying PCA need normal scaling 
    '''   
    # Applying PCA on removed outliers data #
    from sklearn.decomposition import PCA
    pca = PCA(len(input_data.T))
    #Scaling thing for input dataset
    from sklearn.preprocessing import scale
    scld_input_data= scale(input_data, axis=0, with_mean=True, with_std=True, copy=True )
    pca.fit(scld_input_data)
    #Transforming input dataset into the new PC's axis
    pca_input_data=pca.transform(scld_input_data)
    # Expalnied variance realted to each principla compenntes#
    pca.explained_variance_ratio_
    print '\nVariance explanied by eigenvalues of PCA'
    print (['PCA'])
    Var_explanied=np.c_[pca.explained_variance_ratio_]
    print Var_explanied
    return pca_input_data

def perform_kpca(input_data):
    '''
    Applying Kernal PCA on removed outliers data#
    Using scikit module for Kpca
    '''
    from sklearn.decomposition import KernelPCA
    
    # Specify  kernal fucntion  used in the K pca
    KERNEL = raw_input('Enter the kernal of kernalPCA(options are :cosine,rbf,linear,sigmoid:')
    kpca=KernelPCA(n_components=len(input_data.T),kernel=KERNEL)
    #Scaling thing for input dataset
    from sklearn.preprocessing import scale
    scld_input_data= scale(input_data, axis=0, with_mean=True, with_std=True, copy=True )
    kpca.fit(scld_input_data)
    # Transform the dataset on the given PC's
    kpca_input_data=kpca.transform(scld_input_data)
    #Percentage variance representarion
    Kpca_percent=np.array(map(lambda y: (kpca.lambdas_[y]/sum(kpca.lambdas_)),range(len(kpca.lambdas_))))
    Var_explanied=np.c_[Kpca_percent.reshape(len(Kpca_percent),1)]
    print '\nVariance explanied by eigenvalues of KPca '
    print (['Kpca'])
    print Var_explanied          
    return (kpca_input_data)


