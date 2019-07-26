import model_class as mc
import protein_class as pc

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

        self.protein_concentration = protein_concentration

        self.name = name
    
    def new_DNA(self,x):
        