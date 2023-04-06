#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

  echo "*************************************************"
  echo "Produce a fake MinBias event library"
  cmsRun ${SCRAM_TEST_PATH}/testFakeMinBias_cfg.py || die "cmsRun testFakeMinBias_cfg.py" $?

  echo "*************************************************"
  echo "Produce a fake pileup library (premix stage1)"
  cmsRun ${SCRAM_TEST_PATH}/testPremixStage1_cfg.py || die "cmsRun testPremixStage1_cfg.py" $?

  echo "*************************************************"
  echo "Test premixing by adjusting the pileup"
  cmsRun ${SCRAM_TEST_PATH}/testPremixStage2_cfg.py || die "cmsRun testPremixStage2_cfg.py" $?
