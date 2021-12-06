from __future__ import print_function
__author__="Aurelija"
__date__ ="$2010-07-26 12.51.12$"

from os.path import join
import pickle
import os

def createPickleFile(fileName, listRule, path = os.getcwd()):

    dict = {}
    
    try:
        for filePath, lines in listRule:
           dict[filePath] = lines
        file = open(join(path, fileName), 'wb')
        pickle.dump(dict, file)
        file.close()
    except TypeError:
        print('Wrong types')
    except IOError:
        print('Cannot open %s file'%fileName)
