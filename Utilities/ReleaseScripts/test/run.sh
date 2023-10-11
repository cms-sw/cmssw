#!/bin/bash -ex
which scram
scram list -c $(echo $CMSSW_VERSION | sed 's|_X.*|_X|')
#testing 
