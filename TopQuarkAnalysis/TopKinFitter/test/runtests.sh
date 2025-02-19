#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/ttFullLepKinSolutionProducer_cfg.py || die 'Failure using ttFullLepKinSolutionProducer_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/ttFullHadKinFitProducer_cfg.py || die 'Failure using ttFullHadKinFitProducer_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/ttSemiLepKinFitProducer_cfg.py || die 'Failure using ttSemiLepKinFitProducer_cfg.py' $?
