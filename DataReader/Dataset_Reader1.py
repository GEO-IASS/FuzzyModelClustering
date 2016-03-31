import numpy as np
def dataset_reader1():    
    np.set_printoptions(precision=4, suppress=True)
    from read_data_python_codes import *    
    input_fileds = ['PREPARED_SAND.gcs', 'PREPARED_SAND.compactibility', 'PREPARED_SAND.activeClay', 'PREPARED_SAND.wetTensileStrength', 'PREPARED_SAND.loi', 'PREPARED_SAND.moisture', 'PREPARED_SAND.inertFines', 'PREPARED_SAND.volatileMatter', 'PREPARED_SAND.specimenWeight', 'PREPARED_SAND.permeability', 'PREPARED_SAND.friabilityIndex', 'PREPARED_SAND.gfnAfs']
    output_fields = ['REJECTION.rejectionQuantity', 'REJECTION.brokenMould', 'REJECTION.erosionScab', 'REJECTION.sandFusion']
    input_output_data = np.loadtxt('C:/Users/rahuly/Google Drive/Python/Python_Files/SandMan/data.txt', "float64")
    input_data, output_datas = db_data_to_matrix(input_output_data, len(input_fileds))
    # required output that contans only total rejetion percentage
    output_data = output_datas[:, 0]    
    from sklearn.utils import shuffle
    input_data, output_data=shuffle(input_data, output_data, random_state=0)
    dataset=np.c_[ input_data,output_data]
    return  dataset