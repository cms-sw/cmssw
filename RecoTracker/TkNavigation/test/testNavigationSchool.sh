 #!/bin/bash -ex
function die { echo $1: status $2 ; exit $2; }
TEST_DIR=$CMSSW_BASE/src/RecoTracker/TkNavigation/test
echo "test dir: $TEST_DIR"

printf "testing navigation school for Run-3 \n\n"
cmsRun ${TEST_DIR}/NavigationSchoolAnalyzer_cfg.py || die "Failure running NavigationSchoolAnalyzer_cfg.py" $?

printf "testing navigation school for Phase-2 \n\n"
cmsRun ${TEST_DIR}/NavigationSchoolAnalyzer_Phase2_cfg.py || die "Failure running NavigationSchoolAnalyzer_Phase2_cfg.py" $?
