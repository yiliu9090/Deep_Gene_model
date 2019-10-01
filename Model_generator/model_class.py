import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
from tf.keras.models import Model,load_model
from tf.keras.layers import Dense, Dropout, Flatten,Conv1D,Conv2D, AveragePooling1D,Input,Multiply,Add, Lambda,Subtract,ZeroPadding2D,multiply,maximum,Maximum
from tf.keras.optimizers import SGD,Adadelta
from tf.keras.constraints import Constraint, non_neg,NonNeg
from tf.keras.initializers import RandomUniform,Constant
from tf.keras import regularizers
from tf.keras import backend as K

class Positive_lambda(Constraint):

    def __init__(self, max_value=5,min_value = 0.5, axis=0):
        self.max_value = max_value
        self.min_value = min_value
        self.axis = axis

    def __call__(self, w):

        w *= K.cast(K.greater_equal(w, self.min_value), K.floatx())
        norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
        desired = K.clip(norms, 0, self.max_value)
        w *= (desired / (K.epsilon() + norms))
        return w

    def get_config(self):
        return {'max_value': self.max_value,
                'axis': self.axis}

'''
First Model class for the initialization of the result
'''

class DNA_protein_block(tf.keras.Model): 
    '''
    DNA_protein_block should take in different PWM and give out models that allows for PWM to 
    
    take place i.e. something like a model that can be used for further construction
    
    This is a giant building block that allows the user to build on in the future. 
    
    
    
    '''
    def __init__(self, PWM , PWMrc ,max_s, step_size,  concen_input = Input(shape=(None,1,1)),\
                 DNA = Input(shape=(None,4,1)) , score_cut = 0,\
                 adjustment = 0, name = 'Nothing'):
        
        '''
        PWM and PWMrc are in the form 
        
        K.expand_dims(K.expand_dims( K.variable(value=PWM, dtype='float64', name='Kernel'),-1),-1)
        
        max_s is floating point number
        
        concen_input must be of the form Input(shape=(None,1,1))
        
        DNA must be of the form Input(shape=(None,4,1))
        
        score_cut_off must be a number which represents the number of which we want to 
        score to be cut off at 0
        
        step_size is the number of the size of the PWM (how many base pairs)
        
        adjustment is a special adjustment that allows the 
        
        '''
        
        self.PWM = PWM 
        
        self.PWMrc  = PWMrc
        
        self.max_s = max_s
        
        self.concen_input = concen_input
        
        self.DNA = DNA
        
        self.score_cut = score_cut
        
        self.step_size = step_size
        
        self.adjustment = adjustment
        
        
    def call(self):
        
        PMW_Score   = tf.nn.conv2d(self.DNA, self.PWM, strides=[1,1,1,1], padding='VALID')
        
        PMWrc_Score = tf.nn.conv2d(self.DNA, self.PWMrc, strides=[1,1,1,1], padding='VALID')
        
        Indicator_f = K.cast(K.greater(PMW_Score ,self.score_cut),'float32')
        
        Indicator_r = K.cast(K.greater(PMWrc_Score ,self.score_cut),'float32')

        Indicator  = Maximum()([Indicator_r, Indicator_f])
        
        S_relu_f = Lambda(lambda x:K.relu(x, alpha=0.0, max_value=None, threshold=self.score_cut))(PMW_Score)
        
        S_relu_r = Lambda(lambda x:K.relu(x, alpha=0.0, max_value=None, threshold=self.score_cut))(PMWrc_Score)
        
        S_relu  = Maximum()([PMW_Score, PMWrc_Score])
        
        S_i_S_max = Lambda(lambda x: x-self.max_s )(S_relu)
        
        S_i_S_max_lam = Conv2D(1,kernel_size= 1,padding='valid',use_bias=False,kernel_constraint=\
                                   Positive_lambda(),kernel_initializer=RandomUniform(minval=0.25, maxval=2,\
                                                                                      seed=None))(S_i_S_max)
        K_i_m_n = Lambda(lambda x: K.exp(x))(S_i_S_max_lam)
        
        K_relu = Multiply()([K_i_m_n,Indicator] )
        
        Ko_relu = ZeroPadding2D(((0,self.step_size + self.adjustment),(0,self.adjustment)))(K_relu)
        
        return Multiply()([Ko_relu , self.concen_input])
        
        
class DNA_protein_block_loose(tf.keras.Model): 
    '''
    This is a loose computation of the DNA block with many approximation to many numbers 
    
    For one, it does not have exact cut off but instead using a moving number for its cut-offs. 

    However, what we gain from this is cleaner code and faster speed in terms of computation. :) 

    This code will be used for computing human genome.
    
    
    '''
    def __init__(self, PWM , PWMrc ,max_s, step_size,  concen_input = Input(shape=(None,1,1)),\
                 DNA = Input(shape=(None,4,1)) , score_cut = 0,\
                 adjustment = 0, name = 'Nothing'):
        
        '''
        PWM and PWMrc are in the form 
        
        K.expand_dims(K.expand_dims( K.variable(value=PWM, dtype='float64', name='Kernel'),-1),-1)
        
        max_s is floating point number
        
        concen_input must be of the form Input(shape=(None,1,1))
        
        DNA must be of the form Input(shape=(None,4,1))
        
        score_cut_off must be a number which represents the number of which we want to 
        score to be cut off at 0
        
        step_size is the number of the size of the PWM (how many base pairs)
        
        adjustment is a special adjustment that allows the 
        
        '''
        self.PWM = PWM 
        
        self.PWMrc  = PWMrc
        
        self.max_s = max_s
        
        self.concen_input = concen_input
        
        self.DNA = DNA
        
        self.score_cut = score_cut
        
        self.step_size = step_size
        
        self.adjustment = adjustment
        
        
    def call(self):
        
        PMW_Score   = tf.nn.conv2d(self.DNA, self.PWM, strides=[1,1,1,1], padding='VALID')
        
        PMWrc_Score = tf.nn.conv2d(self.DNA, self.PWMrc, strides=[1,1,1,1], padding='VALID')
        
        S_relu  = Maximum()([PMW_Score, PMWrc_Score])
        
        S_i_S_max = Lambda(lambda x: x-self.max_s )(S_relu)
        
        S_i_S_max_lam = Conv2D(1,kernel_size= 1,padding='valid',use_bias=False,kernel_constraint=\
                                   Positive_lambda(),kernel_initializer=RandomUniform(minval=0.25, maxval=2,\
                                                                                      seed=None))(S_i_S_max)
        K_i_m_n = Lambda(lambda x: K.exp(x))(S_i_S_max_lam)

        K_relu  = Conv2D(1,kernel_size= 1,padding='valid',use_bias=True,\
                            activation ='relu' ,\
                            kernel_constraint= NonNeg(),\
                            kernel_initializer=RandomUniform(minval=0.25, maxval=2,seed=None),\
                            bias_initializer=RandomUniform(minval=-2, maxval=0,seed=None))(K_i_m_n)
        
        Ko_relu = ZeroPadding2D(((0,self.step_size + self.adjustment),(0,self.adjustment)))(K_relu)
        
        return Multiply()([Ko_relu , self.concen_input])

#We need to construct the Kenneth Class

class K_algorithm(tf.keras.Model):

    '''
    This is a code for Kenneth's algorithm which is represented as a set of recurrent neural networks.
    '''

    def __init__(self,proteins = {},cooperativity = {},bio_size = {}):

        '''
        We need to have proteins

        We need to have cooperativity  
        '''
        self.proteins = proteins

        self.cooperativity = cooperativity

        self.bio_size = bio_size
    
    def call(self):

        '''
        Need to edit the dynamic programming algorithm to make it work.

        The key is to have a shift in the movement




        '''






