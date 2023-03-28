#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/ttFullLepKinSolutionProducer_cfg.py || die 'Failure using ttFullLepKinSolutionProducer_cfg.py' $?

cmsRun ${SCRAM_TEST_PATH}/ttFullHadKinFitProducer_cfg.py || die 'Failure using ttFullHadKinFitProducer_cfg.py' $?

cmsRun ${SCRAM_TEST_PATH}/ttSemiLepKinFitProducer_cfg.py || die 'Failure using ttSemiLepKinFitProducer_cfg.py' $?
