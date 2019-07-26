import model_class as mc
import protein_class as pc
import numpy as np
'''
This is a DNA_data class structure that allows me to store all the data in a 

DNA_data format. This gives me the luxury to
'''

class Organism_data:
    '''
    DNA data allows to add DNA data and protein data so that they will be 

    accepted by the system 
    '''
    def __init__(self, DNA = 'ACGT', protein_concentration = {},name = 'Unknown'):

        self.DNA = DNA

        self.DNA_value = self.convert()

        self.protein_concentration = protein_concentration

        self.name = name
    
    def new_DNA(self,x, name = 'Unknown'):

        assert(type(x) == type('a'))

        return Organism_data(DNA = x, protein_concentration= \
            self.protein_concentration, name = name)
    
    def new_protein(self,x,name ='Unknown'):

        assert(type(x) == type({}))

        return Organism_data(DNA = self.DNA, protein_concentration= \
            x, name = name)
    
    def update_DNA(self, x):

        assert(type(x) == type('a'))

        self.DNA = x 
        self.DNA_value = self.convert()

    def update_protein(self,x):

        assert(type(x) == type({}))

        self.protein_concentration = x
    
    def convert(self):
        '''
        this function converts DNA sequences into the basic numpy array that
        we are used to 
        '''
        
        n = len(self.DNA)
        
        m = np.zeros((n,4))
    
        count = 0
        
        for i in self.DNA:
        
            if i == 'A' or i == 'a':
                m[count][0] = 1
            elif i == 'C' or i =='c':
                m[count][1] = 1
            elif i == 'G' or i == 'g':
                m[count][2] = 1
            elif i == 'T' or i == 't':
                m[count][3] = 1
            else:
                raise wrong_input
                
            count = count + 1
            
        return(m.reshape((1,n,4,1)))
