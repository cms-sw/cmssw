#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

F1=${LOCAL_TEST_DIR}/python/testG4Geometry.py
F2=${LOCAL_TEST_DIR}/python/testG4Regions.py
F3=${LOCAL_TEST_DIR}/python/testZMM_13TeV_TuneCUETP8M1_cfi_GEN_SIM.py

echo " testing SimG4Core/DD4hepGeometry"

export tmpdir=${LOCAL_TMP_DIR:-/tmp}
echo "===== Test \"cmsRun testG4Geometry.py\" ===="
(cmsRun $F1) || die "Failure using cmsRun $F1" $?
echo "===== Test \"cmsRun testG4Regions.py\" ===="
(cmsRun $F2) || die "Failure using cmsRun $F2" $?
echo "===== Test \"cmsRun testZMM_13TeV_TuneCUETP8M1_cfi_GEN_SIM.py\" ===="
######## FIXME: (cmsRun $F3) || die "Failure using cmsRun $F3" $?
