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
    version 0.1.0
    
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

        self.Protein_concentration = Protein_concentration

        self.name = name_of_organism
        
        self.DNA_length = len(self.DNA)
        
        self.DNA_value = self.convert_DNA()
        
        self.protein_checks()
        
    def protein_checks(self):
        
        '''
        This function checks for 'coherence' of the protein data set. It first checks existence, then checks 
        
        for all the protein the have the same length.
        
        '''
        if self.Protein_concentration == {}: #No cell data at all 
            
            print('Warning: No Cells data') # No cell data 
            
            self.No_of_cells = 0 
            
        else: 
            
            #Check if all the protein_concentration has the same length
            
            cell_len_list = []
            
            for i in self.Protein_concentration:
                
                cell_len_list.append(self.Protein_concentration[i].shape[0])
                
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
    
    def Bases_to_indices(self, base):
        '''
        A simple function to turn bases ACGT to indices 0123
        
        This follows from the concept of simple functions where each function only do one thing
        '''
        if base == 'A' or base == 'a':
            return(0)
        elif base == 'C' or base == 'c':
            return(1)
        elif base == 'G' or base == 'g':
            return(2)
        elif base == 'T' or base == 't':
            return(3)
        else: 
            print('Warning, Bases not of the ACGT')
    
    def convert_DNA(self):
        
        '''
        this function converts DNA sequences into the basic numpy array for training
        '''
        
        m = np.zeros((self.DNA_length ,4))
    
        
        for i in range(self.DNA_length):
            
            m[i][self.Bases_to_indices(self.DNA[i])] = 1 #Bases_to_indices() is suppose to find the index
            
        return(m.reshape((1,1,self.DNA_length,4,1))) #Return tensorflow accepted format
    
    def training_data_build(self, Name_of_target_protein, list_of_tfs, mode = ''):
        '''
        This function builds the training data from the Organism.
        
        The output should look like this 
        
        [[[DNA, TF_concentration1....], protein_level], [[DNA, TF_concentration1....], protein_level]]...]
        
        [DNA, TF_concentration1....]:= X_data 
        
        protein_level:= Y_data 
        
        @param Name_of_target_protein: str of Name of target protein
        @param list_of_tfs: list of str
        
        '''
        
        try:
            self.Protein_concentration[Name_of_target_protein]
        except:
            print('Error, protein not found')
        
        for i in list_of_tfs: #First check that all TF is stored
            
            try: 
                    
                self.Protein_concentration[i]
                    
            except: 
                    
                print('Error, protein not found')
                    
        output_data = []
        
        for cell in range(self.No_of_cells):
        
            X_data = [self.DNA_value] #Dependent variable 
        
            for i in list_of_tfs: 
            
                X_data.append(np.ones((1,1,self.DNA_length,1,1),dtype='float64')*self.Protein_concentration[i][cell])
                #Add the new data to the method
            
            output_data.append([X_data, self.Protein_concentration[Name_of_target_protein][cell] ])
        
        return(output_data)
        
    
    def test_data_build(self,list_of_tfs, mode = ''):
        '''
        This function builds the test data from the Organism
        
        The output should look like this 
        
        [[[DNA, TF_concentration1....], protein_level], [[DNA, TF_concentration1....], protein_level]]...]
        
        [DNA, TF_concentration1....]:= X_data 
        
        protein_level:= Y_data 
        
        @param list_of_tfs: list of str
        
        '''
        for i in list_of_tfs: #First check that all TF is stored
            
            try: 
                    
                self.Protein_concentration[i]
                    
            except: 
                    
                print('Error, protein not found')
                    
        output_data = []
        
        for cell in range(self.No_of_cells):
        
            X_data = [self.DNA_value] #Dependent variable 
        
            for i in list_of_tfs: 
            
                X_data.append(np.ones((1,1,self.DNA_length,1,1),dtype='float64')*self.Protein_concentration[i][cell])
                #Add the new data to the method
            
            output_data.append(X_data)
        
        return(output_data)