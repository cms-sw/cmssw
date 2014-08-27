#!/bin/tcsh

setenv pwd $PWD

# prompt analysis directory
setenv CAF_TRIGGER /afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRIGGER
setenv L1CMSSW CMSSW_3_8_1_patch4

# GRID UI
source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.csh

# setup CMSSW environment
cd $CAF_TRIGGER/l1analysis/cmssw/$L1CMSSW/src

eval `scram ru -csh`
cd $pwd

# CRAB
#source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.csh
source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.csh

# include scripts in path
setenv  PATH ${PATH}:$CAF_TRIGGER/l1analysis/scripts

# add jobs to pythonpath
setenv PYTHONPATH ${PYTHONPATH}:$CAF_TRIGGER/l1analysis/cmssw/$L1CMSSW/src/UserCode/L1TriggerDPG/test

# add ROOT macros to ROOTPATH - doesn't work, grrr
setenv ROOTPATH ${ROOTPATH}:$CAF_TRIGGER/l1analysis/macros

if ( ! -e "$CAF_TRIGGER/$USER/ntuples" ) then
   ln -s $CAF_TRIGGER/l1analysis/ntuples $CAF_TRIGGER/$USER/ntuples
endif
if ( ! -e "$CAF_TRIGGER/$USER/macros" ) then
   ln -s $CAF_TRIGGER/l1analysis/macros $CAF_TRIGGER/$USER/macros
endif

