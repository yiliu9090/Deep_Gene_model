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


'''
First, I need to build a K-cell. Which does one step kenneth algorithm in one direction I will compute this will extremely 

careful system so that things works.


'''

class K_cell(tf.keras.layers.Layer):

    '''
    This is the code for one step 
    error:
    ValueError: Tensor's shape (100, 9, 1) is not compatible with supplied shape [100, 1, 1]
    '''

    def __init__(self,\
                 cooperativity_initial_matrix,\
                 cooperativity_range_matrix, \
                 padding_add_shapes,combine_cooperativity, \
                 non_cooperativity_matrix,\
                 **kwargs):

        '''
        We need to have proteins

        We need to have cooperativity 
        
        @param cooperativity_initial_matrix is a (Number of Cooperativity relationships)x(Number of TF) matrix. tf.constant
            The operation is 
            (cooperativity_range_matrix x current_TF_concentration) = cooperativity_TF
            It aims to get the TFs. Suppose that for one TF there is more than 1 cooperativitiy relationship,
            then matrix will have 2 rows with the same column index
            
        @param cooperativity_range_matrix is a (Number of Cooperativity relationships)x(times step saved) matrix tf.constant
                the key use of this matrix is narrow down to the right sites of TF concentration 
                cooperativity_TF * cooperativity_range_matrix = TF_multiply
        
        @param padding_add_shapes is a truple with (Number of Cooperativity relationships),(Biological footprint of cooperation)
        
        @param combine_cooperativity is a (number of TF with coopertivity)x(Cooperativity relationships) merge relationships
            with the TFs 
        
        @param non_cooperativity_matrix is a matrix (number of TF)x(number of Z saved) backtracking the Zs
        
        @param saved_cooperativity_transformation (Number of cooperativity)x(Number of TF) matrix, ensure that the cooperativity
                saved is going to match the 
        
        '''
        self.cooperativity_initial_matrix = cooperativity_initial_matrix
        self.Num_of_cooperativities = tf.shape(cooperativity_initial_matrix).numpy()[0]
        
        self.cooperativity_range_matrix = cooperativity_range_matrix
        
        self.combine_cooperativity = combine_cooperativity
        self.add_shapes = padding_add_shapes
        
        self.non_cooperativity_matrix = non_cooperativity_matrix
        self.state_size = [tf.TensorShape((None,)),tf.TensorShape((None,))]
        self.output_size = [tf.TensorShape((9,1)),tf.TensorShape((9,1))]
        
        super(K_cell, self).__init__(**kwargs)
    
    
    def build(self,input_shape):
        
        il = input_shape
        '''
        We need to figure out what 
         # expect input_shape to contain 2 items, [(batch, i1), (batch, i2, i3)]
        
        '''
        
        
        self.kernel = self.add_weight(name='kernel',shape=(self.Num_of_cooperativities,1),
                                          initializer=RandomUniform(minval=0, maxval=5, seed=None),
                                          constraint = NonNeg(),
                                          trainable=True,
                                          dtype='float64')


    
    def call(self,inputs, states):
        # inputs should be in [(batch, bcd, cad, ....)]
        # state should be in shape [(batch, Zs), (batch, bcd) (batch, cad) ....]
        '''
        There are two states that is outputed [(batch, Z), (batch, saved concentration in a matrix form)]
        '''
        Z = states[0]
        
        TF_concentrations_saved = states[1]
        
        '''
        Inputwise, there is only input which is a list 
        '''
        current_TF_concentration = inputs[0]
        
        '''
        Start computing Z Cooperativity ***This feature has not fully realize yet***
        
        
        '''
        print(self.cooperativity_initial_matrix)
        cooperativity_TF = tf.linalg.matmul(self.cooperativity_initial_matrix, current_TF_concentration)
        
        '''
        The matrix self.cooperativity_initial_matrix should be a tf.constant() matrix. 
        
        the matrix is (Number of Cooperativity relationships)x(times step saved)
        
        There will be (number of cooperativity relationships) rows and (number of TF) columns where
        each row has one 1 and (number of TF)-1 0's. The position of 1 is exactly at the actor TF position
        
        The output should be a matrix with (number of cooperativity relationships) rows and 1 column
        '''
        TF_multiply = tf.math.multiply(TF_concentrations_saved, self.cooperativity_range_matrix)
        
        '''
        Now, we add some zeros to the front of the cooperativity range matrix to ensure that it will correspond to the 
        right value on the Z axis.
        '''
        TF_multiply_add = tf.concat([tf.zeros(self.add_shapes,dtype =tf.dtypes.float64 ), TF_multiply],1 )
        
        '''
        ***Currently, this means that the TF actor that have cooperativity must be of same size *** 
        ***since different size will mean that some of rows have to be rolled ***
        ***For that case, write your own ***
        At the moment this is true. Because we only deal with bcd
        In the long....long future, it might not be true. 
        As of 8/15/2020, I have sent a request to tensorflow team for help... lets see how it will be dealt with...
        '''
        '''
        Matrix multiply to get q_(i-j) * Z_{i -(j-m)} use notation in algorithm 1 of 
        https://academic.oup.com/bioinformatics/article/36/Supplement_1/i499/5870526
        '''
        TF_multiply_Z = tf.linalg.matmul(TF_multiply_add,Z)
        
        '''
        q_(i)*q_(i-j) * Z_{(i -j-m)}
        '''
        Zc_individual_cooperativity = self.kernel*cooperativity_TF*TF_multiply_Z
        '''
        In case that a TF cooperatively binds to multiple TF.
        '''
        print(Zc_individual_cooperativity)
        Zc_by_TF = tf.linalg.matmul(self.combine_cooperativity, Zc_individual_cooperativity)
        
        '''
        Compute Zc 
        '''
        Zc_sum =tf.math.reduce_sum(Zc_by_TF,keepdims=True)
        
        ################################################## Cooperativity Done #################################
        '''
        Start computing Z non-cooperativity
        '''
        
        '''
        Here we compute the backward Z in this case
        '''
        Z_nc_past = tf.linalg.matmul(self.non_cooperativity_matrix,Z)
     
        '''
        Compute TF concentration of the past
        '''
        
        Znc_by_TF = current_TF_concentration*Z_nc_past
        
        

        '''
        Sum it up to get Z non_cooperativity 
        '''
        Z_nc_sum =tf.math.reduce_sum(Znc_by_TF,keepdims=True)
        
        #####################################non-cooperativity is done! ###############################
        
        '''
        Compute Z
        '''
        Z_new = Z[-1,-1] + Zc_sum +  Z_nc_sum
        '''
        output (Zc, Znc)
        '''
        
        output = (Zc_by_TF,Znc_by_TF)
        
        '''
        compute TF 
        '''
        Z = K.concatenate([Z_new, tf.transpose(Z[:-1])])
        
        Z = tf.transpose(Z)
        '''
        
        '''

        TF_concentrations_new = tf.concat([cooperativity_TF, TF_concentrations_saved[:,:-1]],1)
        #update Z and update TF_concentration_saved
        state = (Z, TF_concentrations_new) #continued
        Zc_by_TF = tf.expand_dims(Zc_by_TF,0)
        Znc_by_TF = tf.expand_dims(Znc_by_TF,0)
        output = (Zc_by_TF,Znc_by_TF)

        
        return (output, state)
    
    
    def get_config(self):
        
        return {}
    