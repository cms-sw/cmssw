#!/bin/bash -ex
which scram
ls -l
scram list -c $(echo $CMSSW_VERSION | sed 's|_X.*|_X|')

