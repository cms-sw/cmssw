#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

(cmsRun ${SCRAM_TEST_PATH}/writeFile_phase2_cfg.py 0) || die 'Failure running cmsRun writeFile_phase2_cfg.py 0' $?
