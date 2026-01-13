#!/bin/bash -ex
which scram
ls -l
whoami
scram list -c $(echo $CMSSW_VERSION | sed 's|_X.*|_X|')

