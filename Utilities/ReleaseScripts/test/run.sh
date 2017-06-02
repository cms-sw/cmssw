#!/bin/bash -ex
echo $CMS_PATH
which scram
scram list -c $(echo $CMSSW_VERSION | sed 's|_X.*|_X|')

