import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv1D,Conv2D, AveragePooling1D,Input,Multiply,Add, Lambda,Subtract,ZeroPadding2D,multiply,maximum,Maximum
from tensorflow.keras.optimizers import SGD,Adadelta
from tensorflow.keras.constraints import Constraint, non_neg,NonNeg
from tensorflow.keras.initializers import RandomUniform,Constant
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

class short_range_interaction_constraint(Constraint):

    def __init__(self, max_value=5,min_value = 0.5):
        self.max_value = max_value
        self.min_value = min_value

    def __call__(self, w):
        w = tf.clip_by_value(w, self.min_value, self.max_value, name=None)
        return w

    def get_config(self):
        return {'max_value': self.max_value, 'min_value':self.min_value}





class TF_short_range_interactions(tf.keras.layers.Layer):
    '''
    build quenching and cooperativity
    '''
    def __init__(self,interaction_type, interaction_kernel,\
                 actor_indices, acted_index,**kwargs):
        '''
        
        '''
        self.interaction_type = interaction_type #accepts 'quenching', 'coactivation'
        self.interaction_kernel = interaction_kernel #for convolution 
        self.padding_left_right = (tf.shape(interaction_kernel).numpy()[0]-1)/2
        self.actor_indices =  tf.constant(actor_indices) # (a,b) a beginning and b ending 
        self.actors_size = len(actor_indices)
        self.acted_index = tf.constant(acted_index) #one index 
        super(TF_short_range_interactions, self).__init__(**kwargs)
        
    def build(self,input_shapes):
        '''
        We need to figure out what 
        '''
        
        self.kernel = self.add_weight(name='kernel',shape=(1,1,self.actors_size,1),
                                          initializer=RandomUniform(minval=0, maxval=1, seed=None),
                                          constraint = short_range_interaction_constraint(max_value=1,min_value = 0.0),
                                          trainable=True,
                                          dtype='float64')
        
    def call(self, inputs):
        '''
        f_{new} =  f_{B} exp(- sum d_i log(1-E_A f_A))
        '''
        actors = tf.gather(inputs,indices = self.actor_indices , axis = 2)
        
        acted  = tf.gather(inputs,indices = self.acted_index ,axis = 2)

        ef_A = tf.math.log( 1.0 - self.kernel*actors) 
        
        ef_kernel = tf.nn.conv2d(
                ef_A, self.interaction_kernel,\
                strides = 1, padding =[[0, 0], [self.padding_left_right,self.padding_left_right], [0, 0], [0, 0]],\
                data_format='NHWC', dilations=None, name=None
                )
        ef_acted = acted*(1-tf.math.exp(ef_kernel))
        
        return ef_acted
    
    def get_config(self):
        
        return { 'Interaction_type':self.interaction_type, #accepts 'quenching', 'coactivation'
                 'Interactions_kernel':self.interaction_kernel, #for convolution 
                 'padding':self.padding_left_right,
                 'actor_indices':self.actor_indices, # (a,b) a beginning and b ending 
                 'actors_size':self.actors_size,
                 'acted_index':self.acted_index}
        
    
class TF_long_range_interactions(tf.keras.layers.Layer):
    '''
    activation and direct repression
    '''
    def __init__(self,interaction_type,\
                 actor_indices, sign, **kwargs):
        '''
        
        '''
        self.interaction_type = interaction_type #accepts 'quenching', 'coactivation'
        self.actor_indices =  tf.constant(actor_indices) # (a,b) a beginning and b ending 
        self.actors_size = len(actor_indices)
        self.sign = sign
        super(TF_long_range_interactions, self).__init__(**kwargs)
        
        
    def build(self,input_shapes):
        
        self.kernel = self.add_weight(name='kernel',shape=(1,1,self.actors_size,1),
                                          initializer=RandomUniform(minval=0, maxval=1, seed=None),
                                          trainable=True, constraint = NonNeg(),
                                          dtype='float64')
        self.kernel2 = self.add_weight(name='kernel2',shape=(1,1),
                                          initializer=RandomUniform(minval=-1, maxval=0, seed=None),
                                          trainable=True,
                                          dtype='float64')
        
    def call(self, inputs):
        '''
        f_{new} =  f_{B} exp(- sum d_i log(1-E_A f_A))
        '''
        
        actors = tf.gather(inputs,indices = self.actor_indices , axis = 2)

        ef_A = self.sign*tf.math.reduce_sum(self.kernel*actors, axis=1) + self.kernel2
        
        return ef_A
        
    def get_config(self):
        
        return { 'Interaction_type':self.interaction_type, #accepts 'quenching', 'coactivation'
                 'actor_indices':self.actor_indices, # (a,b) a beginning and b ending 
                 'actors_size':self.actors_size,
                 'sign':self.sign}

class Enhancer_competition(tf.keras.layers.Layer):
    '''
    activation and direct repression
    '''
    def __init__(self,interaction_type,\
                 actor_indices, sign, alpha, **kwargs):
        '''
        
        '''
        self.interaction_type = interaction_type #accepts 'Enhancer_competition'
        self.actor_indices =  tf.constant(actor_indices) # (a,b) a beginning and b ending 
        self.actors_size = len(actor_indices)
        self.sign = tf.constant(sign) 
        self.alpha = alpha
        self.cnn_kernel = tf.constant(np.ones((self.alpha,1,1,1)))
        super(Enhancer_competition, self).__init__(**kwargs)
        
        
    def build(self,input_shapes):
        
        self.kernel1 = self.add_weight(name='kernel',shape=(1,1,self.actors_size,1),
                                          initializer=RandomUniform(minval=0, maxval=1, seed=None),
                                          trainable=True, constraint = NonNeg(),
                                          dtype='float64')
        self.kernel2 = self.add_weight(name='bias',shape=(1,),
                                          initializer=RandomUniform(minval=0, maxval=1, seed=None),
                                          trainable=True, constraint = NonNeg(),
                                          dtype='float64')
        self.kernel3 = self.add_weight(name='beta',shape=(1,),
                                          initializer=RandomUniform(minval=0, maxval=1, seed=None),
                                          trainable=True, constraint = NonNeg(),
                                          dtype='float64')
    def call(self, inputs):
        '''
        f_{new} =  f_{B} exp(- sum d_i log(1-E_A f_A))
        '''
        
        actors = tf.gather(inputs,indices = self.actor_indices , axis = 2)

        N = tf.math.reduce_sum(self.sign*self.kernel1*actors, axis=2, keepdims=True)
        
        N_cnn = tf.nn.conv2d(N,self.cnn_kernel, strides = 1, padding =[[0, 0], [0,0], [0, 0], [0, 0]],\
                data_format='NHWC', dilations=None, name=None)
        
        R_malpha = tf.math.sigmoid(N_cnn - self.kernel2)
        
        T_malpha = self.kernel3*N_cnn
        
        T = T_malpha/(1 + K.sum(T_malpha))
        
        R_total = K.sum(T*R_malpha,keepdims=True) 
        
        return R_total
    
    def get_config(self):
        
        return { 'Interaction_type':self.interaction_type, #accepts 'quenching', 'coactivation'
                 'Interactions_kernel':self.interaction_kernel, #for convolution 
                 'padding':self.padding_left_right,
                 'actor_indices':self.actor_indices, # (a,b) a beginning and b ending 
                 'actors_size':self.actors_size,
                 'sign':self.sign, 
                 'alpha':self.alpha,
                 'cnn_kernel':self.cnn_kernel}