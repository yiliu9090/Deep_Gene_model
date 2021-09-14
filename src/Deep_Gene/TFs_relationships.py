import copy
class TF_TF_relationship:
    
    '''
    TF_TF_relationship is a class to record one TF_TF relationship inside the system.
    
    If it is quencher, it must have some method to quench with. Quencher has to be nearby. 
    Long range repressor... 
    Three possiblities of distance function... 
    1. short-range /``\
    2. short-range oscillating pattern   
    3. q_d = 1. ```` Manu. 
    4. ... More but we do not know.....
    
    Roles that an TF can take.
    1. Cooperativity (can happen between proteins of different kind... (but do not put in))
    2. Activator 
    3. Quencher (It can be turned into an activator via co-activation)
    4. Repressor (long range repressors/ short range repressors)
    5. Co-activators ()
    6. Activated Quencher cannot quench itself
    7. No role at all ... 
    8. Many more roles. 
    
    '''
    
    def __init__(self, actors, acted, rtype, output_name =None,input_to = None,properties = {}, name = None):
        self.actors = actors 
        self.acted = acted
        self.rtype = rtype
        self.name  = name
        self.properties = properties 
        self.output_name = output_name
        self.input_to = input_to #another relationships
        self.default_output_name()
        
    def default_output_name(self):
        if self.output_name is None:
            self.output_name = self.rtype + '_' + self.acted
            
    def next_relationships(self, relationships):
        self.input_to = relationships
        
    def update_actor(self, new_actors):
        self.actors = new_actors
        
    def update_acted(self, new_acteds):
        self.acteds = new_acteds
        
    def update_type(self, new_type):
        self.rtype = new_type
        
    def update_property(self, new_property):
        self.properties = new_property
        
    def add_functional_pointer(self, functional, **kwargs):
        self.functional= functional(**kwargs)

class TF_TF_relationship_list:
    
    '''
    This is a few graphs of relationships, every node should point to a new TF relationship
    '''
    def __init__(self, relationships =[], name_of_organism= 'None'):
        
        self.relationships = relationships #list of graphs 
        self.name = name_of_organism #
        self.num_of_relationships = len(relationships)
        self.property_pointer = { #use to change property quickly\ 
            'cooperativity': {'range':60},\
            'activation': {},\
            'repression':{},\
            'quenching':  {'range':100},\
            'coactivation':{'range':150}}  
        self.reorder()
        '''
        #This is not public generally but will be improve upon in each updates The reason to use this is to help set up
        a default relationship pointer.
        '''  
    def update_property_pointer(self, new_property_pointer):
        
        self.property_pointer = new_property_pointer
        self.update_property()
        
    def update_property(self):
        
        for i in self.relationships: 
            if i.rtype in self.property_pointer.keys():
                i.update_property(self.property_pointer[i.rtype])
    def reorder(self):
        '''
        BFS
        '''
        explored_list = [] 
        exploring_list = self.find_initial_relationship()
        
        while len(exploring_list) !=0:
            current_node = exploring_list.pop(0)
            if type(current_node.input_to)== type([]):
                for i in current_node.input_to:
                    if i not in explored_list and i not in exploring_list:
                        exploring_list = exploring_list +[i]
            elif current_node.input_to == None:
                exploring_list = exploring_list
            else:
                if current_node.input_to not in explored_list and current_node.input_to not in exploring_list:
                    exploring_list = exploring_list +[current_node]
            
            explored_list = explored_list + [current_node]
            
        self.relationships = explored_list
    
    def find_initial_relationship(self):
        
        '''
        Find initial relationships, that means relationships without any pointers
        '''
        self.protein_build_list = []
        #find starting point.
        initial = [] #initial pointers
        for i in self.relationships:
            intialized_pointer = True
            for j in self.relationships: 
                if type(j.input_to) == type([]):
                    for k in j.input_to:
                        if k == i:
                            intialized_pointer  = False
                else:
                    if j.input_to ==i:
                        intialized_pointer = False
            if intialized_pointer:
                initial = initial + [i]
        
        return initial
        
    def intersection(self, lst1, lst2): 
        #Intersections 
        return list(set(lst1) & set(lst2)) 