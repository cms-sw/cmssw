#!/bin/bash
function die { echo $1: status $2; exit $2; }
REMOTE="/store/group/phys_tracking/cmssw_unittests/"
DQMFILE="DQM_V0001_R000000001__RelValTTbar_14TeV__CMSSW_12_1_0_pre5-121X_mcRun3_2021_realistic_v15-v1__DQMIO.root"
COMMMAND=`xrdfs cms-xrd-global.cern.ch locate ${REMOTE}${DQMFILE}`
STATUS=$?
echo "xrdfs command status = "$STATUS
if [ $STATUS -eq 0 ]; then
    echo "Using file ${DQMFILE}. Running in ${LOCAL_TEST_DIR}."
    xrdcp root://cms-xrd-global.cern.ch/${REMOTE}${DQMFILE} .
    (makeTrackValidationPlots.py ./${DQMFILE}) || die 'failed running makeTrackValidationPlots.py $DQMFILE' $?
    rm -fr ./${DQMFILE}
else 
  die "SKIPPING test, file ${DQMFILE} not found" 0
fi
