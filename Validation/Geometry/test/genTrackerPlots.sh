#!/bin/bash -ex

VGEO_DIR=$CMSSW_BASE/src/Validation/Geometry
TEST_DIR=${VGEO_DIR}/test/materialBudgetTrackerPlots

if [ ! -d $TEST_DIR ]; then
    mkdir $TEST_DIR && cd $TEST_DIR
else
    rm -rf $TEST_DIR && cd $TEST_DIR
fi

cmsRun ${VGEO_DIR}/test/single_neutrino_cfg.py nEvents=1000 >$TEST_DIR/single_neutrino_cfg.log 2>&1

for geom in {'Extended2015','Extended2017Plan1'}; do
    cmsRun ${VGEO_DIR}/test/runP_Tracker_cfg.py geom=$geom label=Tracker >$TEST_DIR/runP_Tracker_cfg.log 2>&1
done

python ${VGEO_DIR}/test/MaterialBudget.py -s -d Tracker

if [[ ! -z ${JENKINS_UPLOAD_DIR} ]] ; then
   mkdir ${JENKINS_UPLOAD_DIR}/materialBudgetTrackerPlots
   cp  ${TEST_DIR}/Images/*.png ${JENKINS_UPLOAD_DIR}/materialBudgetTrackerPlots/
   cp  ${TEST_DIR}/*.log ${JENKINS_UPLOAD_DIR}/materialBudgetTrackerPlots/ 
fi
