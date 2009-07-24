#!/usr/bin/env python

import os

def selection():

    if os.environ['E_SELECTION'] == 'efromW':
        result = ["drop *", "keep+ pdgId = 24", "keep+ pdgId = -24", "drop pdgId !=11 && pdgId !=-11"]
    elif os.environ['E_SELECTION'] == 'efromZ':
        result = ["drop *", "keep+ pdgId = 23", "drop pdgId !=11 && pdgId !=-11"]
    elif os.environ['E_SELECTION'] == 'efromb':
        result = ["drop *", "keep+ abs(pdgId)>=500 & abs(pdgId)<600", "drop pdgId !=11 && pdgId !=-11"]
    else:
        result = ["drop *"," keep pdgId = {e-}", "keep pdgId = {e+}"]

    return result

if __name__ == "__main__":
    for statement in selection():
        print statement
