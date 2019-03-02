import os
import shutil

def emptyDirectory(path):
    if os.path.exists(path): # if it exist already
        shutil.rmtree(path)
    os.makedirs(path)