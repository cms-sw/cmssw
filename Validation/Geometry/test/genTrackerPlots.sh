#!/bin/bash -ex

VGEO_DIR=$CMSSW_BASE/src/Validation/Geometry

cmsRun ${VGEO_DIR}/test/single_neutrino_cfg.py nEvents=1000 >$LOCALRT/single_neutrino_cfg.log 2>&1
    
#Remove big plugin paths
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v /biglib/$SCRAM_ARCH | tr '\n' ':' | sed 's|:$||')

for geom in {'Extended2015','Extended2017Plan1'}; do
    cmsRun ${VGEO_DIR}/test/runP_Tracker_cfg.py geom=$geom label=Tracker >$LOCALRT/runP_Tracker_cfg.log 2>&1
done

python ${VGEO_DIR}/test/MaterialBudget.py -s -d Tracker

if [[ ! -z ${JENKINS_UPLOAD_DIR} ]] ; then
   mkdir ${JENKINS_UPLOAD_DIR}/materialBudgetTrackerPlots
   cp  ${VGEO_DIR}/Images/*.png ${JENKINS_UPLOAD_DIR}/materialBudgetTrackerPlots/
fi
