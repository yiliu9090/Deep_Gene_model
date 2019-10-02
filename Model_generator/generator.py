import model_class as mc
import protein_class as pc

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model,load_model

from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv1D,Conv2D, AveragePooling1D,Input,Multiply,Add, Lambda,Subtract,ZeroPadding2D,multiply,maximum,Maximum
from tensorflow.keras.optimizers import SGD,Adadelta
from tensorflow.keras.constraints import Constraint, non_neg,NonNeg
from tensorflow.keras.initializers import RandomUniform,Constant
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

class Organism_models:
    '''
    This is a organism model structure with functions
    It is designed to help with training and storing of data in the system so that things are well
    tuned 
    
    organism should be able to chunck out models or at least submodels that we can use for future 
    training 
    
    It also take everything a build it as giant model that does not require much time and code
    
    This makes the model extremely useful biologist to test and use.
    
    This model is completely run on Python 3 and tensorflow 2.0 
    
    '''
    def __init__(self, data =[],protein ={}
                ,target= 'RNA' ,cooperativity = {},protein_interactions ={}
                ,cut_off = {}, trained_model = None,name='Nothing'):   
        
        self.data_storage = data #Need to consider all the storage problem 
        
        self.protein = protein #protein is a dictionary of protein with name as the key and the class of protein as the item
        
        self.target = target #target is the name of 'RNA'
        
        self.protein_interactions = protein_interactions #protein interations is a dictionary of the following form 
        '''
        {'activators':{proteins:['bcd','cad'],range:[150]},'quenchers':{['Hb'],range:[150]},'coactivators':{'bcd':'Hb','cad':'Hb','range':[150]},'coquencher':{}}

        At the moment, only activators, quenchers, coactivators and coquenchers are accepted.
        '''
        self.cooperativity = cooperativity #p
        
        self.cut_off = cut_off
        
        self.trained_model = trained_model
        
        self.name = name
        
        self.constructed_model = False 
    
    '''
    This is data preparations which essentially creates data for the model it

    generates
    '''
    def check(self):
        '''
        This code is make sure that things are checked for errors.
        '''
        assert(type(self.protein) == type({}))
        
        pass 
    
    def protein_checks(self):

        pass

    def training_creation(self):
        pass 

    def prediction_creation(self):
        pass
    '''
    Model Creation and calibration
    '''
    
    def add_update_protein(self,x):

        '''
        This is to add or update a protein
        '''
        assert(type(x) == type(pc.protein()))

        self.protein[x.name] = x
        
        self.cut_off[x.name] = 0 #intialize as such
    
    def del_protein(self,x):

        '''
        This is to delete a protein 
        '''
        assert(x in self.protein)

        del self.protein[x]
        
        
    def q_model_generation(self, stacked = True, adjusting = False):
        '''
        This function construct a sequence of model that is just one above the Kenneth Layer
        
        The model will be simple construction and a resultant model will be uncompiled
        
        It is up to the user on how to use this model from then on
        
        '''
        assert(len(self.protein)>0)

        DNA_input = Input(shape = (None,4,1),name = 'DNA_input')

        concentration_input = []

        results = []

        if not adjusting:
            '''
            allow adjustment is later project
            '''
            adjust={}

        count = 0
        
        for prot in self.protein:

            if not adjusting:
                '''
                allow adjustment is later project
                '''
                adjust[prot] =0
            

            concentration_input  = concentration_input +\
                                     [Input(shape = (None,4,1),name = prot+'_Input')]

            results = results + [self.protein[prot].block_generate(DNA_input,\
                                concentration_input[count],score_cut =self.cut_off[prot],\
                                adjustment =adjust[prot]).call()]
            count += 1
        
        Inputs  = [DNA_input] + concentration_input 

        if stacked:
            
             results = K.stack(results)

        basic_model = Model(inputs=Inputs, outputs=results)

        return(basic_model)
    
    def interaction_model(self):

        '''
        Later work for the last part of CNN structure 
        '''
        inter_model = 0

        return(inter_model)

    def full_model(self):
        
        '''
        Important later work
        '''
        pass 
        
