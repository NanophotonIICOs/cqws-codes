import json 
import numpy as np
import scipy.constants as pc
import os
from pathlib import Path
from cqws.db import db_path

# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# db_path = os.path.join(root_dir, 'db', 'db.json')

class DB:
    def __init__(self,material):
        with open(db_path, 'r') as f:
            self.data_base = json.load(f)
        
        self.material = self.data_base[material]
        
    @property        
    def get_material(self):
        return self.material
    
    def  Eg(self,x,T):
        try:
            egap = eval(self.material["Eg"],{'x':x,'T':T})
            return egap
        except Exception as e:
             raise Exception(f"This material: {self.material:s} have an error!")
            
    def me(self,x):
        try:
            effmass = eval(self.material["me"],{'x':x})
            return effmass
        except Exception as e:
             raise Exception(f"This material: {self.material:s} have an error!")
        
        
    # def material_eqn(self,material,eqn_type):
    #     eqn = base_datos[material][eqn_type]
    #     eqn =  eval(eqn_type,{str(constant):constant,})