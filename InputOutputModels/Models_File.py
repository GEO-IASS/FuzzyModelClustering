import numpy as np

#==========Linear Regression Model=================# 
#=================================================================================================================#
def linear_model(input_data,output_data,k_value,rmse_graph):
    print '============================================================================='  
    print 'Linear Regression Results:'
    folds=k_value
    #Cross validation for linear regression
    from sklearn import cross_validation
    kf = cross_validation.KFold(len(input_data),shuffle=True, n_folds=k_value)
    totRMSE_Reg=[]
    # List to store all the linear regression parameters 
    clf1=map(lambda x: 0,range(folds))
    predict_store=[]
    actualy_store=[]
    i=0
    for train_index, test_index in kf:                
        X_train, X_test = input_data[train_index], input_data[test_index]
        y_train, y_test = output_data[train_index], output_data[test_index]
        # model calling for each algo continous
        #Linear regression
        from numpy.linalg import inv
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        X1 = sm.add_constant(X_train)
        model2 = sm.OLS(y_train, X1)
        clf1[i]=model2.fit()
        import statsmodels.regression.linear_model as srl
        result=srl.RegressionResults(model2,clf1[i].params)
        dot_pr=np.dot(sm.add_constant(X_test) ,result.params)
        predict_store.append(dot_pr.tolist())      
        actualy_store.append(output_data[test_index].tolist()) 
        RMSE=(np.sum((np.dot(sm.add_constant(X_test) ,result.params)-y_test)**2)/X_test.shape[0])**.5             
        totRMSE_Reg.append(RMSE)
        i=i+1    
    RMSE_min=min(totRMSE_Reg)
    RMSE_min_index=totRMSE_Reg.index(RMSE_min)
    summary=raw_input('Need the summary of Linear Regression then give "y" or "Y":') 
    if (summary=='y' or summary=='Y'):
        clf=clf1[RMSE_min_index]
        print clf.summary()
        #Homodescadisiticy check tests related to linear regression#
        import statsmodels.stats.diagnostic as ssd
        test1 = ssd.het_breushpagan(clf.resid, clf.model.exog)
        test1_stat=(['LM stat: ','p-val LM test: ','f-stat(error var,not depend on x):','p-value for the f-stat :'])
        Zip_result=np.array(zip(test1_stat,test1))
        print '============================================================================='
        print 'Breush-pagan:\n', Zip_result.reshape(len( Zip_result),2)
    print '============================================================================='    
    print 'Avg. RMSE_Regression for Given K_value Folds',sum(totRMSE_Reg)/kf.n_folds
    if (rmse_graph==1):
        #graphical represenation
        predict_store1= [item for sublist in predict_store for item in sublist] 
        actualy_store1= [item for sublist in actualy_store for item in sublist] 
        #graph values of y_predicted and actualy
        import matplotlib.pyplot as plt
        indices=range(len(actualy_store1))
        plt.plot(indices,predict_store1,'yo-')
        plt.hold(True)
        plt.text(2,19, r'Blue=Actual,Yellow=Predicted')
        plt.plot( indices,actualy_store1,'bo-')
        plt.title('Actual Vs Predcited Graph')
        plt.ylabel('Target variables')
        plt.xlabel('No. of Datasets')
        plt.show()
#=================================================================================================================#


#=======================Classifier Models====================#
#==Logistics Regression==#
#=================================================================================================================#
def Logistics_model(input_data,label_output_data,output_data,edges_interval,K_value,rmse_graph):
    from sklearn import cross_validation
    from sklearn.linear_model import LogisticRegression
    predict_store=[]
    actualy_store=[] 
    # fucntion to use logistics on the later results 
    def Logistics_results(X_train,y_train,X_test,label_output_data,output_data,edges_interval,test_index):     
        model_lr=LogisticRegression()
        #fitting the model
        model_lr.fit(X_train, y_train)
        predicted_class_Probability= model_lr.predict_proba(X_test)
        #print edges_interval[no_classes-1],predicted_class_Probability.shape
        #print predicted_class_Probability[:,no_classes-1] 
        RMSE_lr=np.array(map(lambda x: edges_interval[x]*predicted_class_Probability[:,x],range(len(edges_interval)))).T
        predicted_value=np.sum(RMSE_lr,axis=1) 
        predict_store.append(predicted_value.tolist())      
        actualy_store.append(output_data[test_index].tolist())     
        #RMSE resluts
        RMSE=(np.sum((np.sum(RMSE_lr,axis=1)-output_data[test_index])**2)/output_data[test_index].shape[0])**(.5)       
        return RMSE

    #Cross validation for logistics classifier    
    kf = cross_validation.KFold(len(input_data),shuffle=True,n_folds=K_value)
    totRMSE_log=[]
    for train_index, test_index in kf:                
        X_train, X_test = input_data[train_index], input_data[test_index]
        y_train= label_output_data[train_index]
        RMSE=Logistics_results(X_train,y_train,X_test,label_output_data,output_data,edges_interval,test_index)      
        totRMSE_log.append(RMSE)
    Avg_RMSE=sum(totRMSE_log)/kf.n_folds       
    if (rmse_graph==1):
        #graphical represenation
        predict_store1= [item for sublist in predict_store for item in sublist] 
        actualy_store1= [item for sublist in actualy_store for item in sublist] 
        #graph values of y_predicted and actualy
        import matplotlib.pyplot as plt
        indices=range(len(actualy_store1))
        plt.plot(indices,predict_store1,'yo-')
        plt.hold(True)
        plt.text(2,19, r'Blue=Actual,Yellow=Predicted')
        plt.plot( indices,actualy_store1,'bo-')
        plt.title('Actual Vs Predcited Graph')
        plt.ylabel('Target variables')
        plt.xlabel('No. of Datasets')
        plt.show()
    return Avg_RMSE
#=================================================================================================================#



#====SVM Model======#
#=================================================================================================================#
def SVM_model(Kernel_SVM,input_data,label_output_data,output_data,edges_interval,K_value,rmse_graph):
    from sklearn import cross_validation
    from sklearn import svm
    predict_store=[]
    actualy_store=[]
    #==fucntion to use SVM==#
    def SVM_results(X_train,y_train,X_test,label_output_data,output_data,edges_interval,test_index):     
        model_svm=svm.SVC(kernel=Kernel_SVM,probability=True)
        #fitting the model
        model_svm.fit(X_train, y_train)
        predicted_class_Probability= model_svm.predict_proba(X_test)
        #print edges_interval[no_classes-1],predicted_class_Probability.shape
        #print predicted_class_Probability[:,no_classes-1] 
        RMSE_lr=np.array(map(lambda x: edges_interval[x]*predicted_class_Probability[:,x],range(len(edges_interval)))).T
        predicted_value=np.sum(RMSE_lr,axis=1) 
        predict_store.append(predicted_value.tolist())      
        actualy_store.append(output_data[test_index].tolist())     
        #RMSE resluts
        RMSE=(np.sum((np.sum(RMSE_lr,axis=1)-output_data[test_index])**2)/output_data[test_index].shape[0])**(.5)       
        return RMSE

    #Cross Validation for the SVM    
    kf = cross_validation.KFold(len(input_data),shuffle=True,n_folds=K_value)
    totRMSE_svm=[]
    #loop for use inbulit cross validation 
    for train_index, test_index in kf:                
        X_train, X_test = input_data[train_index], input_data[test_index]
        y_train= label_output_data[train_index]
        RMSE=SVM_results(X_train,y_train,X_test,label_output_data,output_data,edges_interval,test_index)      
        totRMSE_svm.append(RMSE)
    #Avergare RMSE for the given number of cross valdaition
    Avg_RMSE=sum(totRMSE_svm)/kf.n_folds
    #Graph plot for as Actual and predicted value 
    if (rmse_graph==1):
        #graphical represenation
        predict_store1= [item for sublist in predict_store for item in sublist] 
        actualy_store1= [item for sublist in actualy_store for item in sublist] 
        #graph values of y_predicted and actualy
        import matplotlib.pyplot as plt
        indices=range(len(actualy_store1))
        plt.plot(indices,predict_store1,'yo-')
        plt.hold(True)
        plt.text(2,19, r'Blue=Actual,Yellow=Predicted')
        plt.plot( indices,actualy_store1,'bo-')
        plt.title('Actual Vs Predcited Graph')
        plt.ylabel('Target variables')
        plt.xlabel('No. of Datasets')
        plt.show()
    return Avg_RMSE
#=================================================================================================================#
