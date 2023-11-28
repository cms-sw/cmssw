#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

(cmsRun ${SCRAM_TEST_PATH}/phase2_digi_reco_pixelntuple_cfg.py 0) || die 'Failure running cmsRun phase2_digi_reco_pixelntuple_cfg.py 0' $?
