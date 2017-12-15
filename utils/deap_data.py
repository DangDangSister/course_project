#import libraries
import cPickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def deap_data(num_of_subjects, num_classes):
    
    
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
        
            subject_labels[subject_labels < 5] = 0
        
            subject_labels[subject_labels >= 5] = 1
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
    
    
    
    
    
    
