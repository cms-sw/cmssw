#!/bin/bash -ex
which scram
# added a line
# dummy line added

scram list -c $(echo $CMSSW_VERSION | sed 's|_X.*|_X|')

