#!/bin/bash -ex

VGEO_DIR=$CMSSW_BASE/src/SimTracker/TrackerMaterialAnalysis/
TEST_DIR=.

cmsRun ${VGEO_DIR}/test/trackingMaterialProducer10GeVNeutrino_ForPhaseII.py nEvents=1000 >$TEST_DIR/producer.log 2>&1
cmsRun ${VGEO_DIR}/test/trackingMaterialAnalyser_ForPhaseII.py >$TEST_DIR/plotter.log 2>&1
cmsRun ${VGEO_DIR}/test/listIds_PhaseII.py >$TEST_DIR/listIds.log 2>&1

FILES=./*.png
HTAGS=""

for i in $FILES
do
  HTAGS+="<hr/>$i<hr/><br/><img src=\"$i\"/><br/>"
done

echo $HTAGS>index.html

if [[ ! -z ${JENKINS_UPLOAD_DIR} ]] ; then
   mkdir ${JENKINS_UPLOAD_DIR}/trackerMaterialAnalysisPlots/
   cp  ${TEST_DIR}/*.png ${JENKINS_UPLOAD_DIR}/trackerMaterialAnalysisPlots/
   cp  ${TEST_DIR}/index.html ${JENKINS_UPLOAD_DIR}/trackerMaterialAnalysisPlots/
   cp  ${TEST_DIR}/*.log ${JENKINS_UPLOAD_DIR}/trackerMaterialAnalysisPlots/
fi
