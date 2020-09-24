import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv1D,Conv2D, AveragePooling1D,Input,Multiply,Add, Lambda,Subtract,ZeroPadding2D,multiply,maximum,Maximum
from tensorflow.keras.optimizers import SGD,Adadelta
from tensorflow.keras.constraints import Constraint, non_neg,NonNeg
from tensorflow.keras.initializers import RandomUniform,Constant
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import DNA_data_class as DDC
import k_algorithm as kalgo 
import Model_PWM as mpwm 
import tf_interaction as tf_int
import TF_PWM as tpwm 
import TFs_relationships as tfr 
tf.keras.backend.set_floatx("float64")

class Organism_models:
    '''
    This is a organism model structure with functions to build the corresponding model
    
    '''
    def __init__(self, proteins, relationships, data, target):   
        
        self.proteins = proteins 
        self.relationships = relationships
        self.data = data
        self.target = target
        self.check_and_standardize_protein()
        self.check_relationships()
        self.cooperativity_checks()
        
    def check_and_standardize_protein(self):
        '''
        Check all the proteins
        
        Standardize the size and shape of the proteins
        '''
        self.check_protein_type()
        self.get_protein_name()
        #self.get_protein_shape()
        #self.standardize_protein() #adjustments 
    
    def check_protein_type(self):
        for i in proteins: 
            if(type(i) != type(tpwm.TF_PWM())):
                raise NameError('not a Protein')
    def get_protein_name(self):
        self.protein_name = []
        for i in self.proteins:
            self.protein_name.append(i.name)
    
    
    '''This signifies all the basics of proteins are checked'''
    
    def check_and_standardize_relationships(self):
        
        self.check_relationships()
        #self.build_tf_tf_relationship_layers()
        
    def check_relationships(self):
        '''
        Technical checks: All the proteins are there
        Biological Checks:
        '''
        assert(type(self.relationships) == type(tfr.TF_TF_relationship_list()))
        self.check_protein_exists()
        self.remove_non_used_protein()
        
        
    def check_protein_exists(self): 
        '''
        This checks that the protein exists in a relationship list
        '''
        protein_name_list = self.protein_name + [self.target]
        
        for i in self.relationships.relationships:
            for j in i.actors:
                if j not in protein_name_list:
                    raise NameError('Need Protein Information '+j)
                    
            if i.acted  not in protein_name_list:
                 raise NameError('Need Protein Information '+ i.acted)
            if type(i.output_name) == type([]):
                for k in i.output_name:
                    protein_name_list.append(k)
            else: 
                protein_name_list.append(i.output_name)
    
    def remove_non_used_protein(self):
        '''
        We may have more protein than used
        '''
        used_protein_position = []
        self.useful_protein =[]
        for i in self.relationships.relationships:
            for j in i.actors:
                if j in self.protein_name:
                    if self.protein_name.index(j) not in used_protein_position:
                        used_protein_position.append(self.protein_name.index(j))
            if type(i.acted) == type([]):
                for j in i.acted: 
                    if j in self.protein_name:
                        if self.protein_name.index(j) not in used_protein_position:
                            used_protein_position.append(self.protein_name.index(j))
            else:
                if i.acted in self.protein_name:
                    if self.protein_name.index(i.acted) not in used_protein_position:
                            used_protein_position.append(self.protein_name.index(i.acted))
        
        for i in used_protein_position:
            self.useful_protein.append(self.proteins[i])
        
        self.get_protein_shape()
        self.standardize_protein()
        self.get_initial_protein_positions()
        
    def get_initial_protein_positions(self):
        self.protein_positions = {}
        n = 0
        for i in self.useful_protein:
            self.protein_positions[i.name] = n 
            n += 1 
            
    def get_protein_shape(self):
        self.protein_footprint = {}
        for i in self.proteins: 
            self.protein_footprint[i.name] = i.footprint
        
    def standardize_protein(self):
        self.protein_adjustments = {}
        min_protein_size = min(list(self.protein_footprint.values())) 
        for i in self.protein_footprint.keys():
            self.protein_adjustments[i] = [0 , self.protein_footprint[i] - min_protein_size]
            
                    
                
    def cooperativity_checks(self):
        
        '''
        This is important as it focuses on cooperativity 
        This require first to gather all the relationships
        '''
        self.check_relationships()
        self.cooperativity_relationships = [] 
        for i in self.relationships.relationships:
            if i.rtype == 'cooperativity':
                self.cooperativity_relationships += [i]
        #build cooperativity_initial_matrix
        
        self.build_cooperativity_initial_matrix()   
        self.compute_max_range_of_cooperativity()
        self.build_cooperativity_initial_matrix()
        self.build_cooperativity_range_matrix()
        self.build_padding_add_shapes()
        self.build_combine_cooperativity()
        self.build_non_cooperativity()
        
        
    def print_cooperativity_matrices(self):
        
        print('cooperativity_initial_matrix')
        print(self.cooperativity_initial_matrix)
        print('cooperativity_range_matrix')
        print(self.cooperativity_range_matrix)
        print('Padding Shapes')
        print(self.padding_shapes)
        print('combine cooperativity')
        print(self.combine_cooperativity)
        print('non cooperativity matrix')
        print(self.non_cooperativity_matrix)
        
    def compute_max_range_of_cooperativity(self):
        '''
        This is to compute the max range 
        '''
        ranges = []
        if self.cooperativity_relationships != []:
            for i in self.cooperativity_relationships:
                ranges.append(i.properties['range'])
        self.max_range = max(ranges)
    
    def build_cooperativity_initial_matrix(self):
        '''
        This is to find which indices are saved. 
        '''
        self.cooperativity_initial_matrix = np.zeros((1, len(self.useful_protein)))
        for i in self.cooperativity_relationships: 
            self.cooperativity_initial_matrix[0][self.protein_positions[i.actors[0]]] = 1
        self.cooperativity_initial_matrix = tf.constant(self.cooperativity_initial_matrix,dtype=tf.dtypes.float64 )
            
    def build_cooperativity_range_matrix(self):
        '''
        This is to build the cooperativity_range_matrix
        note that this method allows actor to be of one size only 
        '''
        p = len(self.cooperativity_relationships)
        
        #pick only one cooperativity relationship
        
        self.cooperativity_max_range =  self.max_range +\
                    self.useful_protein[self.protein_positions[self.cooperativity_relationships[0].actors[0]]].footprint-1
        self.cooperativity_range_matrix = np.zeros((p, self.cooperativity_max_range))
        for i in range(p):
            actor = self.useful_protein[self.protein_positions[self.cooperativity_relationships[0].actors[0]]]
            cooperativity_range = self.cooperativity_relationships[i].properties['range']
            for j in range(actor.footprint-1, cooperativity_range + actor.footprint-1):
                self.cooperativity_range_matrix[i][j] = 1
        self.cooperativity_range_matrix = tf.constant(self.cooperativity_range_matrix,dtype=tf.dtypes.float64 )
    
    def build_padding_add_shapes(self): 
        
        self.padding_shapes = (len(self.cooperativity_relationships),\
                              self.useful_protein[self.protein_positions[self.cooperativity_relationships[0].actors[0]]].footprint-1)

        
    def build_combine_cooperativity(self):
        
        self.combine_cooperativity = np.zeros((len(self.useful_protein),\
                                              len(self.cooperativity_relationships)))
        for i in range(len(self.cooperativity_relationships)):
            
            self.combine_cooperativity[self.protein_positions[self.cooperativity_relationships[i].actors[0]]][i] = 1
        
        self.combine_cooperativity = tf.constant(self.combine_cooperativity,dtype=tf.dtypes.float64 )
        
    def build_non_cooperativity(self):
        self.non_cooperativity_matrix = np.zeros((len(self.useful_protein), 2*self.cooperativity_max_range -self.max_range ))
        
        for i in range(len(self.useful_protein)):
            self.non_cooperativity_matrix[i][self.useful_protein[i].footprint-1] = 1. 
        self.non_cooperativity_matrix = tf.constant(self.non_cooperativity_matrix,dtype=tf.dtypes.float64 )
    
    def build_model(self):
        '''
        Build model code using exec
        
        '''
        self.check_relationships()
        
        
        Code_build = 'tf.keras.backend.set_floatx("float64") \nDNA = tf.keras.layers.Input(shape=(1,None,4,1))\n'
        protein_order = {}
        num_protein_track = 0
        #setup input
        
        #setup PWM layer
        for j in self.useful_protein:
            Code_build += j.name + ' = tf.keras.layers.Input(shape=(1,None,1,1))\n'
            protein_order[j.name] = num_protein_track
            num_protein_track += 1
            PWMlayer_code = 'PWM_' + j.name + ',' ' PWMrc_'+ j.name +' = j.PWM_tensor_generate(dtypes =tf.dtypes.float64)\n'
            PWMlayer_code += 'max_s_'+j.name + '= j.max_log_frequency_compute()'
            exec(PWMlayer_code)
            Code_build += j.name + '_PWM_layer '+'= mpwm.PWM_Layer_strict(PWM_'+j.name+' , PWMrc_'+ j.name +\
            ', max_s_'+j.name + ',0,['+str(self.protein_adjustments[j.name][0])+','+ str(self.protein_adjustments[j.name][1])+'], tf.dtypes.float64)\n'
            Code_build += j.name+'_binding_site_k = '+j.name + '_PWM_layer(DNA)\n'
            Code_build += j.name+'_binding_site = '+j.name + '_binding_site_k*'+j.name+'\n'
        
        #binding data 
        Code_build += 'binding_data = tf.concat(['
        for j in self.useful_protein:
            Code_build += j.name+'_binding_site,'
        
        Code_build = Code_build[:-1]
        Code_build += '],axis = 3, name = "Full_stacked_for_k_algorithm")[:,0,:,:,:]\n'
        
        Code_build += 'binding_data_greater = binding_data + 1.0 - K.cast(K.greater(binding_data ,0),tf.dtypes.float64 )\n'
        
        '''
        Build K-cell 
        '''
        KCell = kalgo.K_cell(self.cooperativity_initial_matrix,\
                 self.cooperativity_range_matrix, \
                 self.padding_shapes,self.combine_cooperativity, \
                 self.non_cooperativity_matrix)
        
        K_RNN = tf.keras.layers.RNN(KCell, return_sequences =True, time_major =False,return_state= True)

        Code_build += 'f_data =K_RNN(inputs =binding_data, initial_state =\
        [tf.ones(('+ str(2*self.cooperativity_max_range -self.max_range )+','+ str(len(self.cooperativity_relationships))+'),dtype = tf.dtypes.float64),\
        tf.zeros(('+ str(len(self.cooperativity_relationships))+','+ str(self.cooperativity_max_range)+'),dtype = tf.dtypes.float64)])\n'
        
        Code_build += 'f_reverse =K_RNN(inputs =tf.reverse(binding_data,[1]), initial_state =\
        [tf.ones(('+ str(2*self.cooperativity_max_range -self.max_range )+','+ str(len(self.cooperativity_relationships))+'),dtype = tf.dtypes.float64),\
        tf.zeros(('+ str(len(self.cooperativity_relationships))+','+ str(self.cooperativity_max_range)+'),dtype = tf.dtypes.float64)])\n'
        
        Code_build += 'f_reverse_n = tf.reverse(f_reverse[0][1],[1])\n'
        Code_build += 'f_reverse_c = tf.reverse(f_reverse[0][0],[1])\n'
        Code_build += 'f_s = (f_data[0][0]*f_reverse_n+ f_data[0][1]*f_reverse_c + f_data[0][1]*f_reverse_n)/(f_data[1][0]*binding_data_greater)\n'
        
        self.protein_position_model = self.protein_positions.copy()
        current_max_value = max(self.protein_position_model.values())
        num_protein_track += 1 
        for i in self.relationships.relationships: 
            if i.rtype != 'cooperativity':
                if i.rtype == 'coactivation':
                    Code_build += i.acted + "range_coactivation ="+ str(i.properties['range']) + "\n"
                    Code_build += i.acted + "coactivation_convolution_kernel = tf.ones(("+i.acted + "range_coactivation*2+1," + str(len(i.actors)) + ",1,1), dtype=tf.dtypes.float64)\n"
                    
                    
                    Code_build += i.acted +'_coactivation = tf_int.TF_short_range_interactions(interaction_type = "Coactivation",' 
                    Code_build += 'interaction_kernel =' + i.acted +'coactivation_convolution_kernel,actor_indices = ['
                    for j in i.actors:
                        Code_build += str(self.protein_position_model[j])+','
                    Code_build = Code_build[:-1] + '],'
                    Code_build += 'acted_index = ['+ str(self.protein_position_model[i.acted]) + '])\n'
                    Code_build += i.acted +'c =' + i.acted +'_coactivation (f_s)\n'
                    Code_build += i.acted +'q = f_s[:,:,' + str(self.protein_position_model[i.acted]) +',:] - '+ i.acted + 'c\n'
                    Code_build += 'f_s = tf.concat([f_s,'+ i.acted+'c' + ',' + i.acted +'q' +'], axis = 2)\n'
                    for j in i.output_name:
                        self.protein_position_model[j] = num_protein_track
                        num_protein_track += 1 
                        
                if i.rtype =='quenching':
                    Code_build += i.acted + "range_quenching ="+ str(i.properties['range']) + "\n"
                    Code_build += i.acted + "quenching_convolution_kernel = tf.ones(("+i.acted + "range_quenching*2+1," + str(len(i.actors)) + ",1,1), dtype=tf.dtypes.float64)\n"
                    
                    
                    Code_build += i.acted +'_quenching = tf_int.TF_short_range_interactions(interaction_type = "Quenching",' 
                    Code_build += 'interaction_kernel =' + i.acted +'quenching_convolution_kernel,actor_indices = ['
                    for j in i.actors:
                        Code_build += str(self.protein_position_model[j])+','
                    Code_build = Code_build[:-1] + '],'
                    Code_build += 'acted_index = ['+ str(self.protein_position_model[i.acted]) + '])\n'
                    Code_build += i.acted +'q ='+ i.acted +'_quenching(f_s)\n'
                    Code_build += 'f_s = tf.concat([f_s,'+ i.acted+'q' +'], axis = 2)\n'

                    self.protein_position_model[i.output_name] = num_protein_track
                    
                    num_protein_track += 1 
                    
                if i.rtype == 'activation':
                    Code_build += 'A_activation =tf_int.TF_long_range_interactions(interaction_type ="Activation",\
                 actor_indices=['
                    for j in i.actors:
                        Code_build += str(self.protein_position_model[j])+','
                    Code_build = Code_build[:-1] + '], sign = 1)\n'
                    Code_build += 'output = tf.math.sigmoid(tf.math.reduce_sum(A_activation(f_s), axis = 1))\n'  
        final_line = 'self.model = tf.keras.Model(inputs=[DNA,'
        for i in self.protein_positions.keys():
            final_line += i + ','
        
        final_line =final_line[:-1] + '], outputs=output)'
        print(final_line)
        Code_build += final_line
        print(Code_build)
        exec(Code_build)