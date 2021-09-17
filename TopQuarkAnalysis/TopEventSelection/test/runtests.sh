#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/ttSemiLepSignalSelMVAComputer_cfg.py || die 'Failure using ttSemiLepSignalSelMVAComputer_cfg.py' $?
