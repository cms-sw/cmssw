#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/ttSemiLepJetCombMVAComputer_cfg.py || die 'Failure using ttSemiLepJetCombMVAComputer_cfg.py' $?
