#Fuzzy Model Clustering for Mult-modal learning:

#About the Algorithm
The parameters of data originating from multiple linear models can be learnt using this algorithm. This algorithm uses prediction error as distance measure. For more information refer the [link](http://link.springer.com/article/10.1007%2Fs12572-012-0058-y#page-1)

#Example Discussed in the Code:

100 Data samples originating from three linear models 

1.  `y=0.8x+2`
2.  `y=0.2x+1`
3.  `y=-1.5x`

were loaded in the input (`s_input.txt`) and output files(`s_output.txt`) 

Now goal is to estimate the three model parameters using the 100 Data Samples.

The hyperparameter in this algorithm is the `Nth Number of clusters` ( A guess of maximum number of linear model the data can come from). For eg: If `Nth Number of clusters` is 5. The algorithm computes RMSE assuming Number of models are 1,2,3 upto 5 separately. Finally a plot of Number of models vs RMSE will be plotted. The number of Models and the parameters can be picked by the least value of RMSE. For this data, you can notice that the RMSE drops at 3. Therefore the number of model=3 and the corresponding values of parameters can be noted.

#Dependencies:
- python 2.7
- numpy
- pandas
- scipy
- matplotlib
