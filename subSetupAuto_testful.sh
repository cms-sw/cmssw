#!/bin/bash

#source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh
eval `scramv1 runtime -sh`
source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.sh
source /afs/cern.ch/cms/PPD/PdmV/tools/wmclient/current/etc/wmclient_testful.sh
export PATH=/afs/cern.ch/cms/PPD/PdmV/tools/wmcontrol_testful:${PATH}
export PYTHONPATH=/afs/cern.ch/cms/PPD/PdmV/tools/wmcontrol_testful:${PYTHONPATH}
#cat $HOME/private/$USER.txt | voms-proxy-init -voms cms -pwstdin
export X509_USER_PROXY=` voms-proxy-info -path`
