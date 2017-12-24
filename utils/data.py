#import libraries

import cPickle
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import OneHotEncoder, scale

def load_deap(num_of_subjects, num_classes):


    # load data of a subject
    def load_subject_data(subj_num, num_classes):

        # Subject data directory address
        data_url = '../data_preprocessed_python/s'


        # Subject data file address
        subj_url = data_url + str(subj_num).zfill(2) + '.dat'


        # Load subject data file
        subj_dict = cPickle.load(open(subj_url, 'rb'))


        subj_data = subj_dict['data']
        subj_labels = subj_dict['labels']

        # extract labels for Valence and Arousal

        # Valance and Arousal
        # Two classes
        if num_classes==2:

            subject_labels = subj_labels[:,:2]

            subject_labels[subject_labels <= 5] = 0

            subject_labels[subject_labels > 5] = 1
        # Three classes
        elif num_classes == 3:

            subject_labels = subj_labels[:,:2]

            subject_labels[subject_labels < 4] = 0

            subject_labels[(subject_labels >=4) & (subject_labels <=6)] = 1

            subject_labels[subject_labels > 6] = 2


        # OneHot Encoding

        enc = OneHotEncoder()
        valence = enc.fit_transform(subject_labels[:,0].reshape(-1, 1))
        arousal = enc.fit_transform(subject_labels[:,1].reshape(-1, 1))


        return subj_data, valence, arousal


    num_classes = num_classes

    # list of subjects data
    subject_data_list = []
    valence_list = []
    arousal_list = []

    for subj_num in np.arange(1,num_of_subjects+1):

        subject_stat, valence, arousal = load_subject_data(subj_num=subj_num, num_classes=num_classes)
        subject_data_list.append(subject_stat)
        valence = valence.toarray()
        arousal = arousal.toarray()
        valence_list.append(valence)
        arousal_list.append(arousal)


    return subject_data_list, valence_list, arousal_list








def normalize_features(data_folds, pre_norm=False):

        # Check the pre_norm
  # If pre_norm is True normalize before reducing dimension
  if pre_norm == True:
    # concatenate the data folds
    X = np.concatenate(data_folds, axis=0)

    # Get the shape
    axes_0, axes_1, axes_2 = X.shape

    # Reshape the data
    X = X.reshape(axes_0*axes_1,axes_2 )

    # Mapping features to zero mean and unit variance
    X = scale(X)

    # Reshape to 3darray

    X = X.reshape(axes_0, axes_1, axes_2)

    # Split data into 32 folds
    subject_data_folds = np.split(X, 32, axis=0)

    # Reduce the features dimension
    reduced_data_folds = reduce_dim(subject_data_folds)


    # Concatenate the folds of reduced subject's data
    X = np.concatenate(reduced_data_folds, axis=0)


    # Get the shape
    axes_0, axes_1, axes_2 = X.shape

    # Reshape the data
    X = X.reshape(axes_0, axes_1*axes_2 )

    # Mapping features to zero mean and unit variance
    X = scale(X)

    X = X.reshape(axes_0, axes_1, axes_2)

    # Split data into 32 folds
    reduced_data_folds = np.split(X, 32, axis=0)


  else:

                # Reduce the features dimension
                reduced_data_folds = reduce_dim(data_folds)


                # Concatenate the folds of reduced subject's data
                X = np.concatenate(reduced_data_folds, axis=0)


                # Get the shape
                axes_0, axes_1, axes_2 = X.shape

                # Reshape the data
                X = X.reshape(axes_0, axes_1*axes_2 )

                # Mapping features to zero mean and unit variance
                X = scale(X)

                X = X.reshape(axes_0, axes_1, axes_2)

                # Split data into 32 folds
                reduced_data_folds = np.split(X, 32, axis=0)


  return reduced_data_folds




def reduce_dim(data):

  num_subject = len(data)

  exp, ch, r = data[0].shape
  windows = int(r/10)



  def summerize(batch):

        # Mean of the batch
        batch_stat = batch.mean(axis=-1, keepdims=True)

        # Median of the batch
        batch_stat=np.append(batch_stat, np.median(batch, axis=-1, keepdims=True),axis=-1)

        # Maximum of the batch
        batch_stat=np.append(batch_stat, np.amax(batch, axis=-1, keepdims=True),axis=-1)

        # Minimum of the batch
        batch_stat=np.append(batch_stat, np.amin(batch, axis=-1, keepdims=True),axis=-1)

        # Std of the batch
        batch_stat=np.append(batch_stat, np.std(batch, axis=-1, keepdims=True),axis=-1)


        # Variance of the batch
        batch_stat=np.append(batch_stat, np.var(batch, axis=-1, keepdims=True),axis=-1)



        # Range of the batch

        _range = np.ptp(batch, axis=-1)
        _range = _range.reshape(exp,ch,1)
        batch_stat=np.append(batch_stat, _range, axis=-1)


        # Skewness of the batch
        _skew= skew(batch, axis=-1)
        _skew= _skew.reshape(exp,ch,1)
        batch_stat=np.append(batch_stat, _skew, axis=-1)



        # Kurtosis of the batch
        _kurtosis= kurtosis(batch, axis=-1)
        _kurtosis= _kurtosis.reshape(exp,ch,1)
        batch_stat=np.append(batch_stat, _kurtosis, axis=-1)

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
                  batch = subj_data[:,:,j*windows:]
                  batch_stat = summerize(batch)
                  batch_list.append(batch_stat)
            # Other batches
            else:
                  batch = subj_data[:,:,j*windows:(j+1)*windows]
                  batch_stat = summerize(batch)
                  batch_list.append(batch_stat)
        # Append subject summaries to batch list
        batch_list.append(summerize(subj_data))
        subject_stat = np.concatenate(batch_list, axis=-1)

        # Create matrix of subject number and add it to summary
        subject_num_mat = np.ones((exp,ch,1)) * (i + 1 ) # i + 1 = subject number
        subject_stat = np.append(subject_stat, subject_num_mat, axis=-1)

        # Create matrix of experiments number and add it to summary
        ones_mat = np.ones((exp,ch))
        exp_array = np.arange(1,exp+1)
        exp_mat = ones_mat * exp_array[:,np.newaxis]
        exp_mat = exp_mat.reshape(exp,ch,1)

        # Append number of experiments to subject summary
        subject_stat = np.append(subject_stat, exp_mat, axis=-1)

        # Add subject i to reduced_data list
        reduced_data.append(subject_stat)
  return reduced_data
