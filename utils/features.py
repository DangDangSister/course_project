#import libraries


import numpy as np
from scipy.stats import skew, kurtosis
from sklearn import preprocessing




def train_normalize(data_folds, flag=False):
	
	# Check the flag
	# If flag is True normalize before reducing dimension
	if flag == True:
		# concatenate the data folds
		X = np.concatenate(data_folds, axis=0) # shape (1280, 40, 8064)
		# Get the shape
		axes_0, axes_1, axes_2 = X.shape

		# Reshape the data
		X = X.reshape(axes_0, axes_1*axes_2 ) # shape (1280, 322560)

		# Mapping features to zero mean and unit variance
		X = preprocessing.scale(X)
	
		# Reshape to 3darray

		X = X.reshape(axes_0, axes_1, axes_2)
	
		# Split data into 32 folds
		subject_data_folds = np.split(X, 32, axis=0)

		# Reduce the features dimension
		reduced_data_folds = reduce_dim(subject_data_folds)

		
		# Concatenate the folds of reduced subject's data
		X = np.concatenate(reduced_data_folds, axis=0) # shape (1280, 40, 101)


		# Get the shape
		axes_0, axes_1, axes_2 = X.shape

		# Reshape the data
		X = X.reshape(axes_0, axes_1*axes_2 )

		# Mapping features to zero mean and unit variance
		X = preprocessing.scale(X)

		# Split data into 32 folds
		reduced_data_folds = np.split(X, 32, axis=0)
		

	else:

                # Reduce the features dimension
                reduced_data_folds = reduce_dim(data_folds)
 
                
                # Concatenate the folds of reduced subject's data
                X = np.concatenate(reduced_data_folds, axis=0) # shape (1280, 40, 101)


                # Get the shape
                axes_0, axes_1, axes_2 = X.shape

                # Reshape the data
                X = X.reshape(axes_0, axes_1*axes_2 )

                # Mapping features to zero mean and unit variance
                X = preprocessing.scale(X)

                # Split data into 32 folds
                reduced_data_folds = np.split(X, 32, axis=0)

		
	return reduced_data_folds
		



# reduce vector dimention from 8064D to 4040D
def reduce_dim(data):

	num_subject = len(data)

	

	def summerize(batch):
		# Mean of the batch
    		batch_stat = batch.mean(axis=2, keepdims=True)
    
    		# Median of the batch
    		batch_stat=np.append(batch_stat, np.median(batch, axis=2, keepdims=True),axis=2)
        
    		# Maximum of the batch
    		batch_stat=np.append(batch_stat, np.amax(batch, axis=2, keepdims=True),axis=2)
       
    		# Minimum of the batch
    		batch_stat=np.append(batch_stat, np.amin(batch, axis=2, keepdims=True),axis=2)
        
    		# Std of the batch
    		batch_stat=np.append(batch_stat, np.std(batch, axis=2, keepdims=True),axis=2)
     
        
    		# Variance of the batch
    		batch_stat=np.append(batch_stat, np.var(batch, axis=2, keepdims=True),axis=2)
       
               
        
    		# Range of the batch
    		_range = np.ptp(batch, axis=2)
    		_range = _range.reshape(40,40,1)      
    		batch_stat=np.append(batch_stat, _range, axis=2)
        

    		# Skewness of the batch
    		_skew= skew(batch, axis=2)
    		_skew= _skew.reshape(40,40,1)      
    		batch_stat=np.append(batch_stat, _skew, axis=2)
        
        
        
    		# Kurtosis of the batch
    		_kurtosis= kurtosis(batch, axis=2)
    		_kurtosis= _kurtosis.reshape(40,40,1)      
    		batch_stat=np.append(batch_stat, _kurtosis, axis=2)
        
    		return batch_stat
    


	# iterate on the list of subject data
	reduced_data = []
	for i in range(num_subject):
		
		# Empty list to hold each batch statistical summary
    		batch_list = []

		subj_data = data[i]
    
    		for j in range(10):
        	# Last batch    
        		if j==9:
            			batch = subj_data[:,:,j*806:]
        
            			batch_stat = summerize(batch)
            			batch_list.append(batch_stat)
        
        		# Other batches
        		else:
            			batch = subj_data[:,:,j*806:(j+1)*806]
        
            			batch_stat = summerize(batch)
            			batch_list.append(batch_stat)

    		# Append subject summaries to batch list
    		batch_list.append(summerize(subj_data))        
        
    		subject_stat = np.concatenate(batch_list, axis=2)
    
    		# Create matrix of subject number and add it to summary
    		subject_num_mat = np.ones((40,40,1)) * (i + 1 ) # i + 1 = subject number
    		subject_stat = np.append(subject_stat, subject_num_mat, axis=2)
    
    
    		# Create matrix of experiments number and add it to summary
    		ones_mat = np.ones((40,40))
    		exp_array = np.arange(1,41)
    		exp_mat = ones_mat * exp_array[:,np.newaxis]
    		exp_mat = exp_mat.reshape(40,40,1)

    		# Append number of experiments to subject summary
    		subject_stat = np.append(subject_stat, exp_mat, axis=2)


		# Add subject i to reduced_data list
		reduced_data.append(subject_stat)



	return reduced_data



