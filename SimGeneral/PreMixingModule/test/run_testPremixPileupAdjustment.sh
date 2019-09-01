#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  echo "*************************************************"
  echo "Produce a fake MinBias event library"
  cmsRun ${LOCAL_TEST_DIR}/testFakeMinBias_cfg.py || die "cmsRun testFakeMinBias_cfg.py" $?

  echo "*************************************************"
  echo "Produce a fake pileup library (premix stage1)"
  cmsRun ${LOCAL_TEST_DIR}/testPremixStage1_cfg.py || die "cmsRun testPremixStage1_cfg.py" $?

  echo "*************************************************"
  echo "Test premixing by adjusting the pileup"
  cmsRun ${LOCAL_TEST_DIR}/testPremixStage2_cfg.py || die "cmsRun testPremixStage2_cfg.py" $?

popd

exit 0
