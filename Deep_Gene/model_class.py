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



class K_algorithm(tf.keras.layers.Layer):

    '''
    This is a code for Kenneth's algorithm which is represented as a set of recurrent neural networks.
    '''

    def __init__(self, trainable=True,name='Dynamic_Programming_Layer',dynamic=False,proteins = {},cooperativity = {},bio_size = {}):

        '''
        We need to have proteins

        We need to have cooperativity  
        '''
        self.proteins = proteins

        self.cooperativity = cooperativity

        self.bio_size = bio_size #Different Size 
    
    def build(self):
        self.kernel = [] 
        for i in cooperativity: 
            self.kernel = self.kernel + [self.Variable(initial_value=None,\
                trainable=True,validate_shape=True,caching_device=None,name=None,variable_def=None,dtype=None,import_scope=None,constraint=None)]


    
    def call(self,input):

        def steps(inputs,states):
            Zs = states[0]
            bcds = states[1]
            
            nc = K.reshape(inputs[1:10],(1,9))
            c = inputs[0][0]*self.kernel*K.sum(bcds[15:75,]*Zs[29:89,])
            znow = (K.sum(nc,1)+K.sum(c,1)+Zs[0])
            znow = K.sum(nc,1)+Zs[0]
            Zs = K.concatenate([znow, Zs[:-1]])
           
            bcds = K.concatenate([inputs[0], bcds[:-1]])
            
            return K.concatenate([c,nc]), [Zs,bcds] 

        '''
        #Here we try to optimize Kenneth Layer without 
        class Kenneth(tf.keras.layers.Layer):

    def __init__(self, output_dim):
        self.output_dim = output_dim
        super(Kenneth, self).__init__(name='')

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,1),
                                      initializer=RandomUniform(minval=0, maxval=500, seed=None),
                                      constraint = NonNeg(),
                                      trainable=True,
                                     dtype='float32')
        super(Kenneth, self).build(input_shape)

    def call(self, inputs):
        #define everything first and flatten so 1 dimensional array. then they are all one dimensional
        
        bcd = K.flatten(inputs[0])
        cad = K.flatten(inputs[1])
        dst = K.flatten(inputs[2])
        dic = K.flatten(inputs[3])
        hb  = K.flatten(inputs[4])
        kr  = K.flatten(inputs[5])
        kni = K.flatten(inputs[6])
        gt  = K.flatten(inputs[7])
        tll = K.flatten(inputs[8])
        
        data = K.expand_dims(K.stack([bcd,bcd,cad,dst,dic,hb,kr,kni,gt,tll]),-1)
        print(data)
        def steps(inputs,states):
            Zs = states[0]
            bcds = states[1]
            
            nc = K.reshape(inputs[1:10],(1,9))
            c = inputs[0][0]*self.kernel*K.sum(bcds[15:75,]*Zs[29:89,])
            znow = (K.sum(nc,1)+K.sum(c,1)+Zs[0])
            znow = K.sum(nc,1)+Zs[0]
            Zs = K.concatenate([znow, Zs[:-1]])
           
            bcds = K.concatenate([inputs[0], bcds[:-1]])
            
            return K.concatenate([c,nc]), [Zs,bcds] 
            return nc [Zs,bcds]
        initial_states = [K.ones(90), 
                          K.zeros(90)]

        _ , Zc5_3, Z = K.rnn(steps,data,initial_states,go_backwards=False)
        _ , Zc3_5, _ = K.rnn(steps,data,initial_states,go_backwards=True)
        Z_1 = Z[0][0]
       
        bcd = bcd + 1 -K.cast(K.greater(bcd,0.0000000000001),'float64')
        cad = cad + 1 -K.cast(K.greater(cad,0.0000000000001),'float64')
        dst = dst + 1 -K.cast(K.greater(dst,0.0000000000001),'float64')
        dic = dic + 1 -K.cast(K.greater(dic,0.0000000000001),'float64')
        hb  = hb +  1 -K.cast(K.greater(hb ,0.0000000000001),'float64')
        kr  = kr +  1 -K.cast(K.greater(kr ,0.0000000000001),'float64')
        kni = kni + 1 -K.cast(K.greater(kni,0.0000000000001),'float64')
        gt  = gt +  1 -K.cast(K.greater(gt ,0.0000000000001),'float64')
        tll = tll + 1 -K.cast(K.greater(tll,0.0000000000001),'float64')
        Zc3_5 = K.reverse(Zc3_5,1)

        f_bcd = ((Zc5_3[0,:,0]*Zc3_5[0,:,1]+Zc5_3[0,:,1]*Zc3_5[0,:,0]+Zc3_5[0,:,1]*Zc5_3[0,:,1]) /bcd)/Z_1
        f_cad = ((Zc3_5[0,:,2]*Zc5_3[0,:,2]) /cad)/Z_1
        f_dst = ((Zc3_5[0,:,3]*Zc5_3[0,:,3]) /dst)/Z_1
        f_dic = ((Zc3_5[0,:,4]*Zc5_3[0,:,4]) /dic)/Z_1
        f_hb  = ((Zc3_5[0,:,5]*Zc5_3[0,:,5]) /hb)/Z_1
        f_kr  = ((Zc3_5[0,:,6]*Zc5_3[0,:,6]) /kr)/Z_1
        f_kni = ((Zc3_5[0,:,7]*Zc5_3[0,:,7]) /kni)/Z_1
        f_gt  = ((Zc3_5[0,:,8]*Zc5_3[0,:,8]) /gt)/Z_1
        f_tll = ((Zc3_5[0,:,9]*Zc5_3[0,:,9]) /tll)/Z_1
        

        f_bcd = K.clip(f_bcd,0,0.9999)
        f_cad = K.clip(f_cad,0,0.9999)
        f_dst = K.clip(f_dst,0,0.9999)
        f_dic = K.clip(f_dic,0,0.9999)
        f_hb  = K.clip(f_hb ,0,0.9999)
        f_kr  = K.clip(f_kr ,0,0.9999)
        f_kni = K.clip(f_kni,0,0.9999)
        f_gt  = K.clip(f_gt ,0,0.9999)
        f_tll = K.clip(f_tll,0,0.9999)

        f_bcd = K.expand_dims(K.expand_dims(f_bcd,0),-1)
        f_cad = K.expand_dims(K.expand_dims(f_cad,0),-1)
        f_dst = K.expand_dims(K.expand_dims(f_dst,0),-1)
        f_dic = K.expand_dims(K.expand_dims(f_dic,0),-1)
        f_hb  = K.expand_dims(K.expand_dims(f_hb,0),-1)
        f_kr  = K.expand_dims(K.expand_dims(f_kr,0),-1)
        f_kni = K.expand_dims(K.expand_dims(f_kni,0),-1)
        f_gt  = K.expand_dims(K.expand_dims(f_gt,0),-1)
        f_tll = K.expand_dims(K.expand_dims(f_tll,0),-1)
        return [f_bcd,f_cad,f_dst,f_dic,f_hb,f_kr,f_kni,f_gt,f_tll]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        #shape_a, shape_b = input_shape
        return [(1,None,1)]*9
    
def kenneth(inputs, **kwargs):
    """Functional interface to the `Kenneth` layer.
    """
    return Kenneth(1,**kwargs)(inputs)

        '''







