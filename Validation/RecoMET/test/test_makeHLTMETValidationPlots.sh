#!/bin/bash
function die { echo $1: status $2; exit $2; }
LOCAL_TEST_DIR="$PWD"
REMOTE="/store/group/phys_jetmet/cmssw_unittests/"
REDIRECTOR="root://eoscms.cern.ch/" # root://cms-xrd-global.cern.ch
DQMFILE="DQM_RelValQCD_FlatPt_15_3000HS_14_CMSSW_16_0_0_pre4.root"
OUTDIR="."
SCRIPT="makeHLTMETValidationPlots.py"

COMMMAND=`xrdfs ${REDIRECTOR} locate ${REMOTE}${DQMFILE}`
STATUS=$?
echo "xrdfs command status = "$STATUS

if [ $STATUS -eq 0 ]; then
    printf "Using file ${DQMFILE}.\nRunning in ${LOCAL_TEST_DIR}.\n"
    xrdcp ${REDIRECTOR}/${REMOTE}${DQMFILE} .
    (${SCRIPT} --file ./${DQMFILE} --odir ./${OUTDIR} --met hltPFPuppiMETTypeOne) || die 'failed running ${SCRIPT} ${DQMFILE}' $?
    rm -fr ./${DQMFILE}
else 
	die "SKIPPING test, file ${DQMFILE} not found" 0
fi
