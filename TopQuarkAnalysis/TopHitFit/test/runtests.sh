#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/ttSemiLepHitFitProducer_cfg.py || die 'Failure using ttSemiLepHitFitProducer_cfg.py' $?
