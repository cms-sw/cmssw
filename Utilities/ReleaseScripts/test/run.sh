#!/bin/bash -ex
which scram
# added a line
scram list -c $(echo $CMSSW_VERSION | sed 's|_X.*|_X|')

