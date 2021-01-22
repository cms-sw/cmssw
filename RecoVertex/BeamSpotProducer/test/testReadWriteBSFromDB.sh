#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING BeamSpot From DB Read / Write codes ..."

## clean the input db file
if test -f "EarlyCollision.db"; then
    rm -fr EarlyCollision.db
fi

# test write
cmsRun ${LOCAL_TEST_DIR}/write2DB.py inputFile=${LOCAL_TEST_DIR}/EarlyCollision.txt || die "Failure running write2DB.py" $? 
# test read
cmsRun ${LOCAL_TEST_DIR}/readDB.py unitTest=True inputFile=${PWD}/EarlyCollision.db || die "Failure running readDB.py" $? 
