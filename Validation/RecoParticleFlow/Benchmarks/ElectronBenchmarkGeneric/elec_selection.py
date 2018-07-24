#!/usr/bin/env python

from __future__ import print_function
import os

def selection():

    if os.environ['E_SELECTION'] == 'efromW':
        result = ["drop *", "keep+ pdgId = 24", "keep+ pdgId = -24", "drop pdgId !=11 && pdgId !=-11"]
    elif os.environ['E_SELECTION'] == 'efromZ':
        result = ["drop *", "keep+ pdgId = 23", "drop pdgId !=11 && pdgId !=-11"]
    elif os.environ['E_SELECTION'] == 'efromb':
        result = ["drop *", "keep+ abs(pdgId)>=500 & abs(pdgId)<600", "drop pdgId !=11 && pdgId !=-11"]
    elif os.environ['E_SELECTION'] == 'pions':
        result = ["drop *", "keep pdgId = 211", "keep pdgId = -211"]
    else:
        result = ["drop *"," keep pdgId = {e-}", "keep pdgId = {e+}"]
    return result

def deltaR():
    if os.environ['E_SELECTION'] == 'pions':
        result = 0.05
    else:
        result = 0.2
    return result
        
if __name__ == "__main__":
    for statement in selection():
        print(statement)
