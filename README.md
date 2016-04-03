Fuzzy Model Clustering for Mult-modal learning:

The parameters of data originating from multiple linear models can be learnt using Fuzzy Model Clustering.

100 Data samples originating from three linear models 
1) y=0.8x+2
2) y=0.2x+1
3) y=-1.5x
were taken and the input and output data are given to this algorithm as s_input.txt and s_output.txt

(The parameters of multiple linear models are not known in prior, since we are testing the code we are checking if it returns the same parameter)

The initial guess of number of clusters(number of models) is given and the parameters corresponding to clusters are returned as output.

Dependencies:
numpy
panda
scipy
matplotlib
