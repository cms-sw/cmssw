#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING BeamSpot From DB Read / Write codes ..."

## clean the input db file
if test -f "EarlyCollision.db"; then
    rm -fr EarlyCollision.db
fi

# test write
cmsRun ${SCRAM_TEST_PATH}/write2DB.py inputFile=${SCRAM_TEST_PATH}/EarlyCollision.txt || die "Failure running write2DB.py" $?
# test read
cmsRun ${SCRAM_TEST_PATH}/readDB.py unitTest=True inputFile=${PWD}/EarlyCollision.db || die "Failure running readDB.py" $?
