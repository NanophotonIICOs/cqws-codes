import json 
import numpy as np
import scipy.constants as pc
import os
import glob

class DB:
    PATH = os.path.abspath("../db/db.json")
    def __init__(self):
        with open(DB.PATH, 'r') as f:
            self.data_base = json.load(f)
            

    # def material_eqn(self,material,eqn_type):
    #     eqn = base_datos[material][eqn_type]
    #     eqn =  eval(eqn_type,{str(constant):constant,})