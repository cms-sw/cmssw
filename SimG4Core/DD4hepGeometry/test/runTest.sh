#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

F1=${SCRAM_TEST_PATH}/python/testG4Geometry.py
F2=${SCRAM_TEST_PATH}/python/testG4Regions.py

echo " testing SimG4Core/DD4hepGeometry"

export tmpdir=${PWD}
echo "===== Test \"cmsRun testG4Geometry.py\" ===="
(cmsRun $F1) || die "Failure using cmsRun $F1" $?
echo "===== Test \"cmsRun testG4Regions.py\" ===="
(cmsRun $F2) || die "Failure using cmsRun $F2" $?
