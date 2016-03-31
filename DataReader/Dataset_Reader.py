"""
Suggestions:

a) do all the imports in the starting lines before the declaration of first function - i did
b) where is the data.txt file in this folder?? - please put this in this folder itself
c) write a REQUIREMENTS.txt file in this folder and write all the third party modules that one needs to 
   execute the codes in this folder. Along with the module names, also write their versions.
"""

from sklearn.utils import shuffle
import numpy as np
from read_data_python_codes import db_data_to_matrix

class dataReader():
    """
    @precondition: Data should be stored in the file present in "file_path" through numpy.savetxt method

    """
    def __init__(self,file_path,output_included=True,header_present=True,num_input_fields=None,
                 delimiter='\t',shuffle_data=True,dtype="float64"):
        self.file_path = file_path
        self.output_included = output_included
        self.num_input_fields = num_input_fields
        self.delimiter = delimiter
        self.shuffle_data = shuffle_data
        self.header_present = header_present
        self.dtype = dtype
        self.extract_column_names()
        self.read_data()
        
    def extract_column_names(self):
        if self.header_present:
            with open(self.file_path,'r') as file_pointer:
                first_line = file_pointer.readline()
                all_fields = first_line.split(self.delimiter)
                input_fields = all_fields[:self.num_input_fields]
                output_fields = all_fields[self.num_input_fields:]
        else:
            input_fields = []
            output_fields = []
        [self.input_field_names,self.output_field_names] = [input_fields, output_fields]
    
    
    def read_data(self):
        if self.header_present:
            rows_to_skip = 1
        else:
            rows_to_skip = 0
        input_output_data = np.loadtxt(self.file_path, dtype=self.dtype,delimiter=self.delimiter,skiprows=rows_to_skip)
        self.total_data = input_output_data
        [self.input_data, self.output_data] = db_data_to_matrix(input_output_data, self.num_input_fields)
        if self.shuffle_data:
            [self.input_data, self.output_data] = shuffle(self.input_data, self.output_data, random_state=0)
            self.total_data = np.c_[self.input_data, self.output_data]
            
if __name__ == "__main__":
    file_path = "C:\Users\rahuly\Google Drive\Python\Python_Files\rahul_sandman_intern\DataSets\data.txt"
    data_reader_object = dataReader(file_path,output_included=True,header_present=False,num_input_fields=9,
                                    delimiter=None,shuffle_data=False,dtype="float64")
    print data_reader_object.input_data[:1]
    print data_reader_object.output_data[:1]
    
    
    
    
    
    
