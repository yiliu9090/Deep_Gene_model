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
import model_class as mc

class protein:
    '''
    A protein includes a PWM
    
    The protein class should be able to all the log frequency computed, the max value
    
    This helps the user to input protein as 
    '''
    def __init__(self,name = 'Nothing',PWM = np.array([0.25,0.25,0.25,0.25]),\
                 background_frequency = np.array([0.25,0.25,0.25,0.25]),footprint = 1):
        self.name = name
        
        self.PWM = PWM
        
        self.background_frequency = background_frequency
        
        self.footprint = footprint
        
        self.log_frequency_f ,self.log_frequency_r = self.log_frequency_compute()
        
        self.max_log_frequency = self.max_log_frequency_compute()

    
    def log_frequency_compute(self,background = True):

        y = self.PWM

        z = np.zeros((len(y),4))

        c = 0
        for x in y:

            x = x + self.background_frequency

            z[c] = np.log(np.divide(x/np.sum(x),self.background_frequency))

            c = c + 1

        zr = np.copy(z[::-1])

        for i in range(len(zr)):
            
            zr[i][0], zr[i][1],zr[i][2],zr[i][3]=zr[i][3], zr[i][2],zr[i][1],zr[i][0]
        
        return z , zr
    
    def max_log_frequency_compute(self,backgound = True):
        y =  self.PWM

        z = 0
        
        for x in y:
        
            x = x + self.background_frequency
            
            z = z + np.max(np.log(np.divide(x/np.sum(x),self.background_frequency)))

        return(z)

    def block_generate(self, DNA_input, concentration_input,score_cut =0,adjustment =0):
        
        '''
        Protein block generation essentially allow me to generate blocks of data 

        protein
        '''

        PWMf = K.expand_dims(K.expand_dims( K.variable(value=self.log_frequency_f, dtype='float32', name='PWM_'+self.name +'_f'),-1),-1)
        
        PWMr = K.expand_dims(K.expand_dims( K.variable(value=self.log_frequency_f, dtype='float32', name='PWM_'+self.name +'_r'),-1),-1)
        
        return mc.DNA_protein_block( PWMf , PWMr ,self.max_log_frequency, \
                 self.footprint,  concen_input = concentration_input,\
                 DNA = DNA_input , score_cut = score_cut,\
                 adjustment = adjustment, name = self.name +'_protein_block')