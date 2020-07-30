import model_class as mc
import protein_class as pc
import numpy as np
import pickle
'''
This is a DNA_data class structure that allows me to store all the data in a 

DNA_data format. This gives me the luxury to store the data however I want and take them out as I pleased.
'''

class Organism_data:
    
    '''
    Organism_data is the class that is before the final step to the full training data. It is both a function class 
    for specific functions and a data class of storage of information. 

    We need to following functionality:

    1. Store the data 
    2. Save the data under a pickle file structure.
    3. Change the data (add and update) 
    4. Generate Training data for tensorflow to run on 
    5. Allow for generation of pseudo data to compensate for unknown protein that 
    does not exist or not known. This is useful for testing data construction.
    6. Check for the inner coherence of all the data to make sure that all the inner coherence is known.
    '''

    def __init__(self, DNA = 'ACGT', Protein_concentration = {},name_of_organism = 'Unknown'):
        '''
        
        The data will consist of DNA sequence, a dictionary of protein concentrations and the name of the organism.
        
        At the initialization step, we can calculate the how any cells there are and what is the length of the DNA sequence
        
        '''
        self.DNA = DNA

        self.DNA_value = self.convert_DNA()

        self.Protein_concentration = Protein_concentration

        self.name = name_of_organism
        
        self.DNA_length = len(self.DNA)
        
        self.protein_checks()
        
    def protein_checks(self):
        
        '''
        This function checks for 'coherence' of the protein data set. It first checks existence, then checks 
        
        for all the protein the have the same length.
        
        '''
        if Protein_concentration == {}: #No cell data at all 
            
            print('Warning: No Cells data') # No cell data 
            
            self.No_of_cells = 0 
            
        else: 
            
            #Check if all the protein_concentration has the same length
            
            cell_len_list = []
            
            for i in Protein_concentration:
                
                cell_list.append(Protein_concentration[i].shape(0))
                
            if max(cell_len_list)== min(cell_len_list):
                
                self.No_of_cells = min(cell_len_list)
                
            else:
                
                print('Warning: Cell data conflicting!, minimum cell length taken as default')
                
                self.No_of_cells = min(cell_len_list)
                
            
        
    def new_DNA(self,new_DNA, name = 'Unknown'):
        
        '''
        This function build a new Organism_data class with a new DNA sequence
        '''
        
        assert(type(new_DNA) == type('a'))

        return Organism_data(DNA = new_DNA, Protein_concentration= \
            self.protein_concentration, name = name)
    
    def new_protein_concentration(self,new_Protein,name ='Unknown'):
        
        '''
        This function build a new Organism_data class with a new DNA Protein Concentration set
        '''

        assert(type(new_Protein) == type({}))

        return Organism_data(DNA = self.DNA, Protein_concentration= \
            new_Protein, name = name)
    
    def update_DNA(self, new_DNA):

        assert(type(x) == type('a'))

        self.DNA = new_DNA 
        
        self.DNA_value = self.convert()
        
        self.DNA_length = len(self.DNA)

    def update_protein(self,new_protein):

        assert(type(new_protein) == type({}))

        self.Protein_concentration = new_protein
        
        self.protein_checks()
        
    
    def convert_DNA(self):
        
        '''
        this function converts DNA sequences into the basic numpy array for training
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
    
    def training_data_build(Name_of_target_protein, list_of_tfs):
        '''
        This function builds the training data from the Organism.
        '''
        
        pass
    
    def test_data_build(list_of_tfs):
        '''
        This function builds the test data from the Organism
        '''
        
        pass
