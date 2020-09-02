import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv1D,Conv2D, AveragePooling1D,Input,Multiply,Add, Lambda,Subtract,ZeroPadding2D,multiply,maximum,Maximum
from tensorflow.keras.optimizers import SGD,Adadelta
from tensorflow.keras.constraints import Constraint, non_neg,NonNeg
from tensorflow.keras.initializers import RandomUniform,Constant
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

class Positive_lambda(Constraint):

    def __init__(self, max_value=5,min_value = 0.5):
        self.max_value = max_value
        self.min_value = min_value

    def __call__(self, w):

        w = tf.clip_by_value(w, self.min_value, self.max_value, name=None)
        return w

    def get_config(self):
        return {'max_value': self.max_value,
                'axis': self.axis}

'''
First Model class for the initialization of the result
'''

class PWM_Layer_strict(tf.keras.layers.Layer):
    '''
    
    PWM_Layer_strict follows from Liu, et.(2020), The focus is to use this code is to construct with relative accuracy 
    
    the PWM site identification model in Liu, et. (2020)
    '''
    def __init__(self, PWM, PWMrc, max_s,score_cut, adjustment,dtype_now,  **kwargs ):
        
        '''
        This is an interesting problem
        '''
        
        self.PWM  = PWM
        self.PWMrc = PWMrc
        self.max_s =  max_s
        self.score_cut = score_cut
        self.adjustment = adjustment
        self.paddings = tf.constant([[0,0],[0, 0], [self.adjustment[0], self.adjustment[1]], [0, 0],[0, 0]])
        self.dtype_now = dtype_now
        
        super(PWM_Layer_strict, self).__init__( **kwargs )
        
        
    def build(self, input_shapes):
        
        self.kernel = self.add_weight(name='kernel',shape=(1,1,1,1),
                                          initializer=RandomUniform(minval=0, maxval=1, seed=None),
                                          trainable=True,
                                          constraint =Positive_lambda(), 
                                          dtype=self.dtype_now)

    
    def call(self, inputs):
        
        PMW_Score   = tf.nn.conv2d(inputs[0], self.PWM, strides=[1,1,1,1], padding='VALID')
        
        PMWrc_Score = tf.nn.conv2d(inputs[0], self.PWMrc, strides=[1,1,1,1], padding='VALID')
        
        Indicator_f = K.cast(K.greater(PMW_Score ,self.score_cut),self.dtype_now )
        
        Indicator_r = K.cast(K.greater(PMWrc_Score ,self.score_cut),self.dtype_now )

        Indicator  = Maximum()([Indicator_r, Indicator_f])
        
        S_relu  = Maximum()([PMW_Score, PMWrc_Score])
        
        S_i_S_max = S_relu-self.max_s
        
        S_i_S_max_lam =  S_i_S_max*self.kernel
        
        K_i_m_n = tf.math.exp(S_i_S_max_lam)
        
        K_relu = K_i_m_n*Indicator
        
        Ko_relu = tf.pad(K_relu ,self.paddings,'CONSTANT')
        
        return Ko_relu
