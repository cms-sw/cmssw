#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/ttSemiLepJetCombMVAComputer_cfg.py || die 'Failure using ttSemiLepJetCombMVAComputer_cfg.py' $?
