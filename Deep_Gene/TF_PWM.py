import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv1D,Conv2D, AveragePooling1D,Input,Multiply,Add, Lambda,Subtract,ZeroPadding2D,multiply,maximum,Maximum
from tensorflow.keras.optimizers import SGD,Adadelta
from tensorflow.keras.constraints import Constraint, non_neg,NonNeg
from tensorflow.keras.initializers import RandomUniform,Constant
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

class TF_PWM:
    '''
    The true purpose of protein class is to deal with all the calculations concerning PWMs. Thats it!
    
    Calculation of PWM: 
        1. Relative PWM with backgound frequency
        2. Reverse Compliment Build
        3. Footprint padding
        4. Max constant compute
        5. tf.tensor build 
    
    '''
    def __init__(self,name = 'Nothing',PWM = np.array([[0.25,0.25,0.25,0.25]]),\
                 background_frequency = np.array([0.25,0.25,0.25,0.25]), footprint_adjust = [0,0]):
        
        '''
        @param name: a str of 
        @param PWM: a np.array([A,C,G,T])
        @param background_frequency: a np.array([A,C,G,T])
        @param footprint_adjust: List of 2 numbers. 
        '''
        
        self.name = name #name of the TF
        
        self.PWM = PWM #PWM matrix
        
        self.background_frequency = background_frequency #background Frequency of the organism

        self.biological_foot_print = len(self.PWM) #biological footprint is different from the actual footprint
        
        self.footprint_adjust = footprint_adjust
        
        self.footprint = PWM.shape[0] + self.footprint_adjust[0] + self.footprint_adjust[1]
        
        self.log_frequency_f ,self.log_frequency_r = self.log_frequency_compute()
        
        self.max_log_frequency = self.max_log_frequency_compute()

    
    def log_frequency_compute(self,add_background = True):
        
        '''
        From PWM data table to the actual PWM matrix for computation requires many crucial steps that includes a 
        
        an adding operations followed by a sum operation
        
        '''

        log_PWM_matrix = np.zeros((self.footprint,4))

        reverse_complement = np.zeros((self.footprint,4))
        
        reverse_complement_adjust =  self.footprint_adjust[0] - self.footprint_adjust[1]
        for i in range(len(self.PWM)):

            PWM_one_column = self.PWM[i] + self.background_frequency

            log_PWM_matrix[self.footprint_adjust[0] + i] = np.log(np.divide(PWM_one_column/np.sum(PWM_one_column),\
                                                                            self.background_frequency))
            
            #ACGT A==T, C==G, ACGT
            
            reverse_complement[self.footprint -reverse_complement_adjust -i-1] =\
            self.reverse_complement_op(log_PWM_matrix[self.footprint_adjust[0] + i])
        
        return log_PWM_matrix , reverse_complement
    
    def reverse_complement_op(self,DNA):
        '''
        return the reverse complement so that the code is 
        '''
        return np.array([DNA[3], DNA[2], DNA[1], DNA[0]])
    
    def max_log_frequency_compute(self,add_backgound = True):
        
        '''
        return the maximum log_value
        '''
        max_log_value = 0
        
        for PWM_i in self.PWM:
        
            PWM_one_column = PWM_i + self.background_frequency
            
            max_log_value = max_log_value + np.max(np.log(np.divide(PWM_one_column/np.sum(PWM_one_column),\
                                                                    self.background_frequency)))
        return(max_log_value)

    def PWM_tensor_generate(self, dtypes = 'float32'):
        
        '''
        Protein block generation generates 
        '''

        PWMf = tf.expand_dims(tf.expand_dims( tf.convert_to_tensor(value=self.log_frequency_f, \
                                                                   dtype=dtypes, name='PWM_'+self.name +'_f'),-1),-1)
        
        PWMr = tf.expand_dims(tf.expand_dims( tf.convert_to_tensor(value=self.log_frequency_f, dtype=dtypes, \
                                                                   name='PWM_'+self.name +'_r'),-1),-1)
        
                
        return PWMf, PWMr