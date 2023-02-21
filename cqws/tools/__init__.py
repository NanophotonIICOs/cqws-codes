from .plotstyle import load_style
from .constants import constants

import os
def mkdatadir(path=None):
    # create data files folder
    if path is None:
        dir_path=os.getcwd()
    else: 
        dir_path= path
    print("Current PATH:%s"%(dir_path))
    if os.path.isdir(dir_path+'/data'):
        print("The data folder already exist")
    else:
        print("The data folder doesn't exist") 
        os.mkdir(dir_path+'/data')
        print("The data folder was created successfully!")
    save_path=dir_path+'/data'
    return save_path