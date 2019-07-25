import model_class as mc
import protein_class as pc

class Organism:
    '''
    This is a organism model structure with functions
    It is designed to help with training and storing of data in the system so that things are well
    tuned 
    
    organism should be able to chunck out models or at least submodels that we can use for future 
    training 
    
    It also take everything a build it as giant model that does not require much time
    
    This makes the model extremely useful biologist to test and use. 
    
    This model is completely run on Python 3 and tensorflow 2.0 
    
    '''
    def __init__(self, DNA_data = 'ACGT', protein ={} ,concentrations_data = {},\
                 target={},cooperativity = {},protein_interactions ={},\
                 cut_off = {}, trained_model = [] name='Nothing'):
        '''
        DNA_data 
        
        '''
        self.DNA = DNA_data
        
        self.protein = protein
        
        self.concentrations = concentrations
        
        self.target = target 
        
        self.protein_interactions = protein_interactions
        
        self.cooperativity = cooperativity
        
        self.cut_off = cut_off
        
        self.trained_model = trained_model
        
        self.name = name
        
        self.constructed_model = False 
    
    
    def convert(self):
        '''
        this function converts DNA sequences into 
        '''
        
        n = len(self.DNA)
        
        m = np.zeros((n,4))
    
        count = 0
        
        for i in self.DNA:
            try:
                i in 'ACGTNacgtn'
            except wrong_input:
                print('wrong input')
        
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
    
    def add_update_protein(self,x):

        '''
        This is to add or update a protein
        '''
        
        assert(type(x) == type(pc.protein))

        self.protein[x.name] = x
    
    def del_protein(self,x):

        '''
        This is to delete a protein 
        '''
        
        
    def basic_model_generation(self):
        '''
        This function construct a sequence of model that is just one above the Kenneth Layer
        
        The model will be simple construction and a resultant model will be uncompiled
        
        It is up to the user 
        
        '''
        assert(len(protein)>0)
        
        
        return(basic_model)