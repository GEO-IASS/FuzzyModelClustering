# this below code is modified version of the code written by raghu's btp students
# i am including the computation of gd index and modifying the logic of reassigning the clusters
# most of this code is written by sneha,manoj and suraj.
#lines starting with ## are the lines which are commmented by sreeram
#Shriramajayam

############################################ Libraries Imported #############################################################
import random
import numpy
import copy
import pprint

############################################ Defining all the functions #####################################################
#Function to obtain data from the excel file
def load_into_file(path,header=True):
        # input of this function - 'path' is a variable which tells us the location where the data file is located
        # ouput of this function - looks like a matrix (it will be a list of lists)
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


#Function to to choose cluster points randomly. The number of cluster points and the data is given as an input to the fnction
# Also computing the distances dij

def initial_clstr_pts(num_clstr_pts,data):
    data_array = numpy.array(data)
    ##print'Data Array'
    ##print(data_array)
    dim=data_array.shape[1]
    num_data_pts=data_array.shape[0]
    print'Dimension'
    print(dim)
    MaxMin = numpy.zeros((2,dim))
    #print(MaxMin)
    for i in range(0,dim):
        MaxMin[0][i]=max(data_array[:,i])
        MaxMin[1][i]=min(data_array[:,i])
    ##print'The max and the min values in a given column of data'
    ##print(MaxMin)
    clstr_pts = numpy.zeros((num_clstr_pts,dim))
    #print(clstr_pts)
    Rand=numpy.zeros((1,dim))
    for i in range(0,num_clstr_pts):
        for j in range(0,dim):
            Rand[0][j]=random.uniform(MaxMin[1][j],MaxMin[0][j])
        clstr_pts[i]=Rand

    ##print'Cluster Points'
    ##print(clstr_pts)
    return(clstr_pts,num_data_pts,dim)



# Function to compute the initial membership values
def initial_membership(num_clstr_pts,clstr_pts,num_data_pts,data,q):
        clstr_pts=numpy.array(clstr_pts)
        data_array=numpy.array(data)
        #Calculating dij
        dij = numpy.zeros((num_clstr_pts,num_data_pts))
        for i in range(0,num_clstr_pts):
            for j in range(0,num_data_pts):
                a = clstr_pts[i]
                b = data_array[j]
                dist_a_b = numpy.linalg.norm(a-b)
                dij[i][j]=dist_a_b
        mu_old = numpy.zeros((num_clstr_pts,num_data_pts))
        for i in range(0,num_clstr_pts):
            for j in range(0,num_data_pts):
                sum=0
                for k in range(0,num_clstr_pts):
                    sum =sum+ ( 1/(dij[k][j]) )**(2/(q-1))
                sum=1/sum
                mu_old[i][j]= (1/((dij[i][j])**(2/(q-1))))*sum
        return(mu_old)


# Function for printing out data into an excel file
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

#Function to update cluster points
def updating_clstr_pts(num_data_pts,num_clstr_pts,q,data,mu_old,dim):
        data_array = numpy.array(data)
        mu_old = numpy.array(mu_old)
        new_clstr_pts = numpy.zeros((num_clstr_pts,dim))
        for i in range(0, num_clstr_pts):
                sum_num = numpy.zeros((1,dim))
                sum_den =0
                for j in range(0, num_data_pts):
                        sum_num = sum_num + ((mu_old[i][j])**q)*data_array[j]
                        sum_den = sum_den + ((mu_old[i][j])**q)
                new_clstr_pts[i]= sum_num/sum_den
        return(new_clstr_pts)

def compute_gd_index(total_data_points,membership_matrix):
    #this function written/modified by sreeram
    gd_index=0#this should be close to 0
    gd_index2=0#this should be close to 1
    row_cluster_membership_list=[]
    for row in range(total_data_points):
        membership_list=list(membership_matrix[:,row])
        max_membership=max(membership_list)
        max_index=membership_list.index(max_membership)
        row_cluster_membership_list.append(max_membership)
        del membership_list[max_index]
        if len(membership_list)==0:
            gd_index+=0
        else:
            gd_index+=max(membership_list)
    gd_index2=round(float(sum(row_cluster_membership_list)/total_data_points),2)
    #print float(gd_index/total_data_points)
    print 'gd_index2 is '+ str(gd_index2)
    ##plot_histogram(row_cluster_membership_list)
    return round(float(gd_index/total_data_points),2)		
		
def compute_answers(path,num_clusters,fuzzifier,tolerance_value,percent_min_points,debug_mode,reassigning_flag,print_output,output_path,scaling_data_list,scale_down_input,scale_up_output):
    import pprint
    data_in_file,header_cont = load_into_file(path)
    if scale_down_input:
            for row in range(len(data_in_file)): # all these files have this mixed indent because when i have written these files in notepad,,the indent got screwed 
                data_in_file[row]=un_scale_data(data_in_file[row],scaling_data_list,type='list').tolist()
    reassigning_done=0
    numCP=num_clusters
    clstr_pts_new,num_data_pts,dim = initial_clstr_pts(numCP,data_in_file)	
    print'Number of data points'
    print(num_data_pts)
    q=fuzzifier
    tol=tolerance_value
    mu_old=initial_membership(numCP,clstr_pts_new,num_data_pts,data_in_file,q)
    mu_new = numpy.zeros((numCP,num_data_pts))
    mu_new=mu_old
    distance = numpy.ones((numCP,1))
    check = numpy.ones((numCP,num_data_pts))*100
    loop_counter_variable=0
    while 1:	
        loop_counter_variable+=1
	##print loop_counter_variable
        mu_old=mu_new
        clstr_pts_old=clstr_pts_new			
        clstr_pts_new = updating_clstr_pts(num_data_pts,numCP,q,data_in_file,mu_old,dim)
        mu_new=initial_membership(numCP,clstr_pts_new,num_data_pts,data_in_file,q)
        distance = numpy.zeros((numCP,1))
        for j in range(0,numCP):
            a = clstr_pts_new[j]
            b = clstr_pts_old[j]
            dist_a_b = numpy.linalg.norm(a-b)
            distance[j]=dist_a_b
        distance=numpy.array(distance)
        flag=0
        for i in range(0,numCP):
            if distance[i]>tol:
                flag=1
        if flag==0:
            break
    clstr_pts_new=numpy.array(clstr_pts_new)
    mu_new=numpy.array(mu_new)
    if debug_mode:
        print 'checking the reassigning'
        print clstr_pts_new.shape
    if reassigning_flag:
        mu_new,clstr_pts_new=perform_reassigning(clstr_pts_new,mu_new,percent_min_points,data_in_file,fuzzifier,debug_mode)
        print 'after reassigning the structure of cluster points is ' + str(clstr_pts_new.shape)
        if numCP!=int(clstr_pts_new.shape[0]):
            reassigning_done=1
    if debug_mode:
        print distance
        print loop_counter_variable
        pprint.pprint(clstr_pts_new)
        next_iter=raw_input('press enter when you are ready to see final distance results')
    gd_index=compute_gd_index(num_data_pts,mu_new)
    print 'gd_index is '+ str(gd_index)
    if print_output:        
        write_data_and_clusters(data_in_file,mu_new,header_cont,output_path)
        if scale_up_output:
            for i in range(clstr_pts_new.shape[0]):
                clstr_pts_new[i]=scale_data(clstr_pts_new[i],scaling_data_list,type='list')
        write_to_file(output_path,clstr_pts_new,header_cont,append_flag=1)
    else:
        pass            

    return [gd_index,loop_counter_variable,reassigning_done]
	
'sreeram - changed input argument from path to data_in_file'    
def fcm_clusters_answers(data_in_file,num_clusters,fuzzifier,tolerance_value,percent_min_points,debug_mode,reassigning_flag,output_data_in_file,print_output=0,scaling_data_list=[],scale_down_input=0,scale_up_output=0,reject_col_present=0):    
    header_cont = []
    if scale_down_input:
        for row in range(len(data_in_file)):  
            data_in_file[row]=un_scale_data(data_in_file[row],scaling_data_list,type='list').tolist()
    reassigning_done=0
    numCP=num_clusters
    clstr_pts_new,num_data_pts,dim = initial_clstr_pts(numCP,data_in_file)                      ##line 33  
    print'Number of data points'
    print(num_data_pts)
    q=fuzzifier
    tol=tolerance_value
    mu_old=initial_membership(numCP,clstr_pts_new,num_data_pts,data_in_file,q)                  ##line 63  
    mu_new = numpy.zeros((numCP,num_data_pts))
    mu_new=mu_old
    distance = numpy.ones((numCP,1))
    check = numpy.ones((numCP,num_data_pts))*100
    loop_counter_variable=0
    while 1:    
            loop_counter_variable+=1
            ##print loop_counter_variable
            mu_old=mu_new
            clstr_pts_old=clstr_pts_new            
            clstr_pts_new = updating_clstr_pts(num_data_pts,numCP,q,data_in_file,mu_old,dim)   ##line 105
            mu_new=initial_membership(numCP,clstr_pts_new,num_data_pts,data_in_file,q)         ##line 63 
            distance = numpy.zeros((numCP,1))
            for j in range(0,numCP):
                a = clstr_pts_new[j]
                b = clstr_pts_old[j]
                dist_a_b = numpy.linalg.norm(a-b)
                distance[j]=dist_a_b
            distance=numpy.array(distance)
            flag=0
            for i in range(0,numCP):
                if distance[i]>tol:
                    flag=1
            if flag==0:
                break
    clstr_pts=copy.deepcopy(clstr_pts_new)
    clstr_pts_new=numpy.array(clstr_pts_new)
    mu_new=numpy.array(mu_new)
    if debug_mode:
        print 'checking the reassigning'
        print clstr_pts_new.shape
    if reassigning_flag:
        mu_new,clstr_pts_new=perform_reassigning(clstr_pts_new,mu_new,percent_min_points,data_in_file,fuzzifier,debug_mode)
        print 'after reassigning the structure of cluster points is ' + str(clstr_pts_new.shape)
        if numCP!=int(clstr_pts_new.shape[0]):
            reassigning_done=1
    if debug_mode:
        print distance
        print loop_counter_variable
        pprint.pprint(clstr_pts_new)
        next_iter=raw_input('press enter when you are ready to see final distance results')
    gd_index=compute_gd_index(num_data_pts,mu_new)
    print 'gd_index is '+ str(gd_index)
    if scale_up_output:
            for i in range(clstr_pts_new.shape[0]):
                clstr_pts_new[i]=scale_data(clstr_pts_new[i],scaling_data_list,type='list')
            for i in range(len(data_in_file)):
                data_in_file[i]=scale_data(data_in_file[i],scaling_data_list,type='list').tolist()
    if print_output:                
        write_data_and_clusters(data_in_file,mu_new,header_cont,output_path)
        write_to_file(output_path,clstr_pts_new,header_cont,append_flag=1)
    if scale_down_input:# since i have separate flags for scaling down input for clustering and another flag for scaling up cluster centres output
        for i in range(clstr_pts.shape[0]):
                clstr_pts[i]=scale_data(clstr_pts[i],scaling_data_list,type='list')                  
    data_clusters,output_clusters=get_data_clusters(data_in_file,output_data_in_file,mu_new,reject_col_present) #line 391 this line assigns data to clusters (say c1,c2) based on maximum membership
    return mu_new,clstr_pts,data_clusters,output_clusters,header_cont


def write_data_and_clusters(data_in_file,mu_new,header_cont,output_path):
	data_in_file_copy=copy.deepcopy(data_in_file)
	num_cp,pts_to_clstr=compute_clusters_and_distribution(mu_new)
	for row in range(len(data_in_file_copy)):
		cluster='C'+str(int(pts_to_clstr[:,row].tolist().index(1))+1)
		data_in_file_copy[row].append(cluster)
	write_to_file(output_path,data_in_file_copy,header_cont)
		
def compute_clusters_and_distribution(mu_new):
	numCP=mu_new.shape[0]#number of clusters
	num_data_pts=mu_new.shape[1]#number of data points
	#Defining an array which will indicate the number of data points belonging to each cluster point
	pts2clstr=numpy.zeros((numCP,num_data_pts))

	#Checking which cluster point the data point belongs to
	for j in range(0,num_data_pts):
		max = mu_new[0][j]
		index=0
		for i in range(1,numCP):
			if mu_new[i][j]>max:
				max=mu_new[i][j]
				index=i
		pts2clstr[index][j]=1
	return numCP,pts2clstr

def reassigning_indices_order(pts2clstr):
	count_list=[]
	for row in pts2clstr:
		count_list.append(sum(row))
	copy_count_list=copy.deepcopy(count_list)
	reassigning_list=[]
	iteration_count=len(count_list)
	for i in range(iteration_count):
		index=count_list.index(min(copy_count_list))
		reassigning_list.append(index)
		copy_count_list.remove(count_list[index])
	return reassigning_list
	
def perform_reassigning(cluster_centres,mu,min_percent,input_data,q,debug_mode):
	total_data_points=mu.shape[1]
	while 1:
		num_clusters,cluster_distribution=compute_clusters_and_distribution(mu)
		count=0
		iter_list=reassigning_indices_order(cluster_distribution)
		if debug_mode:
			for row in range(num_clusters):
				print 'sum of points belong to cluster '+str(row)
				print row,sum(cluster_distribution[row])			
				print iter_list
		for key in iter_list:
			count_num_points=sum(cluster_distribution[key])
			if count_num_points<int(min_percent*total_data_points/100):
				mu,cluster_centres=reassigning_cluster(cluster_centres,key,input_data,q)
				if debug_mode:
					print ' I am reassigning the points of the cluster '+str(key)+' to other clusters'
					#copy_in_data has to be redefined
					print cluster_centres
					print key
					print cluster_centres
					raw_input('check')
				break
			count+=1
		if count==num_clusters:
			break
		#break
	num_clstr_pts=int(cluster_centres.shape[0])
	mu_new=initial_membership(num_clstr_pts,cluster_centres,total_data_points,input_data,q)
	return mu_new,cluster_centres


def reassigning_cluster(cluster_centres,key,input_data,q):
	#Reallocating data points of discarded cluster points
	#Criteria Used: Each data point belonging to a discarded cluster point is reallocated to the cluster oint closest to it
	cluster_centres_copy=copy.deepcopy(cluster_centres)
	num_clstr_pts=int(cluster_centres_copy.shape[0])-1
	data_array=numpy.array(input_data)
	num_data_pts=data_array.shape[0]
	cluster_centres_copy=numpy.delete(cluster_centres_copy,key,axis=0)
	mu_new=initial_membership(num_clstr_pts,cluster_centres_copy,num_data_pts,input_data,q)
	return mu_new,cluster_centres_copy
	
def un_scale_data(data_clusters,scaling_data_list,type='dict'):
    copy_data=copy.deepcopy(data_clusters)    
    if type=='dict':
        for key in copy_data:
            for row in copy_data[key]:
                for i in range(len(scaling_data_list)):
                    factor=float(scaling_data_list[i][1]-scaling_data_list[i][0])
                    row[i]=(row[i]-float(scaling_data_list[i][0]))/factor
        copy_data[key]=numpy.array(copy_data[key])
    if type=='list':
        for i in range(len(scaling_data_list)):
            factor=float(scaling_data_list[i][1]-scaling_data_list[i][0])            
            copy_data[i]=(float(copy_data[i])-float(scaling_data_list[i][0]))/factor
        copy_data=numpy.array(copy_data)
    return copy_data

def scale_data(data_clusters,scaling_data_list,type='dict'):
    copy_data=copy.deepcopy(data_clusters)    
    if type=='dict':
        for key in copy_data:
            for row in copy_data[key]:
                for i in range(len(scaling_data_list)):
                    factor=float(scaling_data_list[i][1]-scaling_data_list[i][0])
                    row[i]=(row[i]*factor)+float(scaling_data_list[i][0])
            copy_data[key]=numpy.array(copy_data[key])
    if type=='list':
        for i in range(len(scaling_data_list)):
            factor=float(scaling_data_list[i][1]-scaling_data_list[i][0])
            copy_data[i]=(copy_data[i]*factor)+float(scaling_data_list[i][0])
        copy_data=numpy.array(copy_data)
    return copy_data

def get_data_clusters(in_data,output_data,mu,reject_col_present):
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

	
########################################### Executing the functions ##################################
if __name__=='__main__':
        # Calling function to obtain data stored in the excel file      
	path1 = "C:\\Backup\\acads\\7th sem\\Shriramajayam_BTP\\Fuzzy_C_Means\\final_data_set.xls"
	path1="C:\\Users\Suraj\\Desktop\\Project_Assignment\\final_data_set.xls"
	data_in_file = load_into_file(path1)
			
	#Taking the number of cluster points needed as an input from the user
	numCP=input('Please enter the number of cluster points to be chosen initially:')
	print(numCP)

	#Calling the function initial_clstr_pts to randomly choose initial cluster points
	clstr_pts_new,num_data_pts,dim = initial_clstr_pts(numCP,data_in_file[0])

	#writing out the initial cluster points into an excel file
	##path0= "C:\\Backup\\acads\\7th sem\\Shriramajayam_BTP\\Fuzzy_C_Means\\Shriramajayam_My_Python_Codes\\initial_clstr_pts.xls"
	##write_to_file(path0,clstr_pts_new)


	print'Number of data points'
	print(num_data_pts)
	print'Dimension of the data points'
	print(dim)


	#Taking the value of q from the user
	q =     input('Please enter the value of the fuzzifier:')
	print(q)

	#Taking the value of the tolerance from the user
	tol =     input('Please enter the desired tolerance value:')
	print(tol)


	# Calling the function to compute the initial membership values 
	mu_old=initial_membership(numCP,clstr_pts_new,num_data_pts,data_in_file,q)
	##print'Initial membership values'
	##print(mu_old)


	#Code for writing out initial membership values in an excel file
	##path2= "C:\\Backup\\acads\\7th sem\\Shriramajayam_BTP\\Fuzzy_C_Means\\Shriramajayam_My_Python_Codes\\Initial_Membership.xls"
	##write_to_file(path2,mu_old)

	# Initializing mu_new as mu_old
	mu_new = numpy.zeros((numCP,num_data_pts))
	mu_new=mu_old

	# Initializing distance matrix
	distance = numpy.ones((numCP,1))

	#Initializing the value of the matrix check to a matrix with very large entries
	check = numpy.ones((numCP,num_data_pts))*100
	loop_counter_variable=0

	while 1:
	##i=0
	##while i<25:
			loop_counter_variable+=1
			print loop_counter_variable
			mu_old=mu_new
			clstr_pts_old=clstr_pts_new
			
			#Calling the function to update the cluster points
			clstr_pts_new = updating_clstr_pts(num_data_pts,numCP,q,data_in_file,mu_old,dim)
			##print'New Cluster Points'
			##print(clstr_pts_new)
			#writing out the initial cluster points into an excel file
			##path_clstr_new= "C:\\Backup\\acads\\7th sem\\Shriramajayam_BTP\\Fuzzy_C_Means\\Shriramajayam_My_Python_Codes\\new_clstr_pts.xls"
			##write_to_file(path_clstr_new,clstr_pts_new)

			#Function to update membership values i.e calling the initial membership function
			# writing out the answer in a different excel file
			mu_new=initial_membership(numCP,clstr_pts_new,num_data_pts,data_in_file,q)
			##print'New membership values'
			##print(mu_new)

			#Code for writing out new membership values in an excel file
			##path3= "C:\\Backup\\acads\\7th sem\\Shriramajayam_BTP\\Fuzzy_C_Means\\Shriramajayam_My_Python_Codes\\New_Membership.xls"
			##write_to_file(path3,mu_new)

			distance = numpy.zeros((numCP,1))
			for j in range(0,numCP):
				a = clstr_pts_new[j]
				b = clstr_pts_old[j]
				dist_a_b = numpy.linalg.norm(a-b)
				distance[j]=dist_a_b
			
			##print'Distance between subsequent cluster points'
			##print(distance)
			##path_dist= "C:\\Backup\\acads\\7th sem\\Shriramajayam_BTP\\Fuzzy_C_Means\\Shriramajayam_My_Python_Codes\\dist_bet_subsequent_clstr_pts.xls"
			##write_to_file(path_dist,distance)
			distance=numpy.array(distance)

			flag=0
			for i in range(0,numCP):
				if distance[i]>tol:
					flag=1
			if flag==0:
				break
	##        i=i+1
			

	# Printing the value of the final cluster points
	##print'THE FINAL ANSWER'
	##print'FINAL CLUSTER POINTS'
	##print(clstr_pts_new)
	clstr_pts_new=numpy.array(clstr_pts_new)
	#Printing the final membership values
	##print'FINAL MEMBERSHIP VALUES'
	##print(mu_new)
	mu_new=numpy.array(mu_new)
	
	gd_index=compute_gd_index(num_data_pts,mu_new)
	print 'gd_index is '+ str(gd_index)

	#Defining an array which will indicate the number of data points belonging to each cluster point
	pts2clstr=numpy.zeros((numCP,num_data_pts))

	#Checking which cluster point the data point belongs to
	for j in range(0,num_data_pts):
		max = mu_new[0][j]
		index=0
		for i in range(1,numCP):
			if mu_new[i][j]>max:
				max=mu_new[i][j]
				index=i
		pts2clstr[index][j]=pts2clstr[index][j]+1

	#Taking the limit on deciding whether or not the cluster point should be removed
	percent_data_pts_associated =     input('Please enter the percentage of points below which the cluster point should be discarded:')
	print(percent_data_pts_associated)
	##threshold = round(float(percent_data_pts_associated * num_data_pts)/100)
	threshold = round((percent_data_pts_associated * num_data_pts)/100)
	print'Threshold Value'
	print(threshold)
		
	#Code for writing out new membership values in an excel file
	##path_pts2clstr= "C:\\Backup\\acads\\7th sem\\Shriramajayam_BTP\\Fuzzy_C_Means\\Shriramajayam_My_Python_Codes\\pts2clstr.xls"
	##write_to_file(path_pts2clstr,pts2clstr)

	#Finding the number of data points corresonding to eacjh cluster point
	num_data_pts_corr2each_clstr_pt = numpy.zeros((numCP,1))


	#Creating an array with ones and zeros. 1= discarded, 0 = not discarded
	print'creating an array with ones and zeros. 1= discarded, 0 = not discarded'
	discard = numpy.zeros((numCP,1))
	for i in range(0,numCP):
			num_data_pts_corr2each_clstr_pt[i] = sum(pts2clstr[i])
			if num_data_pts_corr2each_clstr_pt[i]<threshold:
					discard[i]=1

	print'Number of data points corresponding to each cluster point'
	print(num_data_pts_corr2each_clstr_pt)
	print'Discarded Points are given by ones'
	print(discard)

	#Finding the indices of all the points which have been discarded
	row_index_discarded=sum(discard)
	row_index_retained=numCP-sum(discard)

	index_discarded = numpy.zeros((row_index_discarded[0],1))
	index_retained = numpy.zeros((row_index_retained[0],1))
	j=0
	for i in range(len(discard)):
			if discard[i]==1:
					index_discarded[j] =i
					j=j+1

	print'The indices of the discarded cluster points'
	print(index_discarded)


	j=0
	for i in range(0,len(discard)):
			if discard[i]==0:
					index_retained[j] =i
					j=j+1

	print'The indices of the retained cluster points'
	print(index_retained)

	#Reallocating data points of discarded cluster points
	#Criteria Used: Each data point belonging to a discarded cluster point is reallocated to the cluster oint closest to it
	data_array=numpy.array(data_in_file)
	for i in range(0,len(index_discarded)):
			a=index_discarded[i]
			num_data_pts_corr2each_clstr_pt[a[0]]=0
			for j in range(0,num_data_pts):
					if (pts2clstr[a[0]][j]==1):
							a1=numpy.zeros((len(index_retained),dim))
							b = data_array[j]
							new=numpy.zeros((len(index_retained),1))
							for k in range(0,len(index_retained)):
									a1[k] = clstr_pts_new[(index_retained[k])[0]]
									new[k]=numpy.linalg.norm(a1-b)
							new_index=numpy.argmin(new)
							pts2clstr[a[0]][j]=0
							pts2clstr[(index_retained[new_index])[0]][j]=1
							num_data_pts_corr2each_clstr_pt[(index_retained[new_index])[0]]=num_data_pts_corr2each_clstr_pt[(index_retained[new_index])[0]]+1
							


	#Prinitng the number of data points corresponding each cluster point after the reallocation
	print'The number of data points corresponding to each cluster point after the reallocation'
	print(num_data_pts_corr2each_clstr_pt)                      


