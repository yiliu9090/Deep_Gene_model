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
    
    def __init__(self, actor, acted, name ,rtype, properties = {}):
        self.actor = actor
        self.acted = acted
        self.rtype = rtype
        self.name  = name
        self.properties = properties 
        
    def update_actor(self, new_actor):
        self.actor = new_actor
        
    def update_acted(self, new_acted):
        self.acted = new_acted
        
    def update_type(self, new_type):
        self.rtype = new_type
        
    def update_property(self, new_property):
        self.properties = new_property
        
    def add_functional_pointer(self, functional):
        self.functional= functional

class TF_TF_relationship_list:
    
    '''
    This is a list of TF, TF relationship. The functions in the this class is to check and conform kind of relationships
    (i.e. no conflicting relationship)
    
    
    
    '''
    def __init__(self, relationships =[], name_of_organism= 'None'):
        
        self.relationships = relationships #all the relationships in a list form
        self.name = name_of_organism #
        self.num_of_relationships = len(relationships)
        self.organize()
        self.check()
        self.property_pointer = {\
            'cooperativity': {'range_max':60, 'range_min':14},\
            'activation': {},\
            'repression':{},\
            'quenching':  {'range_max':150},\
            'coactivation':{'range_max':150}}                         
        '''
        #This is not public generally but will be improve upon in each updates The reason to use this is to help set up
        a default relationship pointer.
        '''
        
        
    def organize(self):
        
        '''
        This is to organize the relationship into different hash tables for easy references.
        '''
         
        self.relationship_types = {}
        self.relationship_actor = {}
        self.relationship_acted = {}
        
        if self.num_of_relationships > 0:
            '''
            Cache all the relationships in hash tables to ensure that we can connect all the things together
            '''
            for i in range(self.num_of_relationships):
                
                rel = self.relationships[i]
                
                if rel.rtype not in self.relationship_types:
                    self.relationship_types[rel.rtype] = [i]
                else: 
                    self.relationship_types[rel.rtype].append(i)
                    
                if rel.actor not in self.relationship_actor:
                    self.relationship_actor[rel.actor] = [i]
                else: 
                    self.relationship_actor[rel.actor].append(i)
                    
                if rel.acted not in self.relationship_acted:
                    self.relationship_acted[rel.acted] = [i]
                else:
                    self.relationship_acted[rel.acted].append(i)
                    
    def delete_TF(self, target):
        '''
        Delete TF deletes all the relationships associated with those TF. 
        '''
        self.delete_actor(target)
        self.delete_acted(target)
        self.organize()
    
        
    def delete_actor(self, target):
        '''
        Delete all the all relationships with target as the actor
        '''
        relationship_target_index = self.relationship_actor[target]
        
        self.relationships.remove(relationship_target_index)
        
        
    
    def delete_acted(self, target):
        '''
        Delete all the all relationships with target as the acted
        '''
        
        relationship_target_index = self.relationship_acted[target]
        
        self.relationships.remove(relationship_target_index)
        
    def add_TF_role(self, TF, target, rtypes):
        '''
        Add a TF role to the relationship stance
        
        Add a class pointer to build a default relationship.
        
        '''
        assert((type(TF) == type([]))\
                    and (type(target)== type([])))
               
        rtype_is_str = True if type(rtypes) == type('r') else  False #A string 
        rtype_is_list = True if (type(rtypes) == type([]))\
                    and (len(rtypes) == len(TF)*len(target)) else False
        
        assert(rtype_is_str or rtype_is_list)
        
        count = 0 #count incase there are multiple 
        
        for i in TF:
            for j in target:
                if rtype_is_list:#rtype is a list  
                    
                    if rtype[count] in self.property_pointer:
                        new_r = TF_TF_relationship(actor = i, acted =j, name = i+j ,rtype =rtype[count],\
                                                   properties =self.property_pointer[rtype[count]])
                        self.add_relationship(new_r)
                       
                    else:

                        new_r = TF_TF_relationship(actor = i, acted =j, name = i+j ,rtype =rtype[count], properties = {})
                        self.add_relationship(new_r)
                       
                    count = count + 1
                else: #rtype is a str
                    if rtypes in self.property_pointer:
                        new_r = TF_TF_relationship(actor = i, acted =j, name = i+j ,rtype =rtypes,\
                                                   properties =self.property_pointer[rtypes])
                        self.add_relationship(new_r)
                       
                    else:
                        new_r = TF_TF_relationship(actor = i, acted =j, name = i+j ,rtype =rtypes, properties = {})
                        self.add_relationship(new_r)
        self.organize()
        self.check()
        
    
    def add_relationship(self, relationship):
        '''
        Add a relationship into the list
        '''
        self.relationships.append(relationship)
        
        self.num_of_relationships += 1
        
    def check(self):
        '''
        
        This is a function to check relationships 
        
        '''
        self.remove_duplicates()
        
    def remove_duplicates(self):
        '''
        
        remove duplicated cases. generally remove only the earlier ones
        
        '''
        
        remove_indx = []
        for i in self.relationship_types:
            
            type_list = self.relationship_types[i]
            
            for j in self.relationship_actor:
                actor_list = self.relationship_actor[j]
                for k in self.relationship_acted:
                    intersections = self.intersection(self.intersection(self.relationship_acted[k],actor_list),type_list)
                    lst_length = len(intersections)
                    if lst_length > 1:
                        remove_indx.append(intersections[:-2])#keep only the last one
        if remove_indx !=[]:
            self.relationships.remove(remove_indx)
        
    def intersection(self, lst1, lst2): 
        #Intersections 
        return list(set(lst1) & set(lst2)) 
    
    def switching_role(self, target, role_initial, role_to): 
        
        '''
        Currently I have no idea how to implement this in a sensible way.
        '''
        allowed_roles = ['quenching', 'coactivation']
        pass
        
    
