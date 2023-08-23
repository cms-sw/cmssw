#!/bin/sh -e

function die { echo $1: status $2 ; exit $2; }
function checkDiff {
    FSIZE=$(stat -c%s "$1")
    echo "The output diff is $FSIZE:"
    cat $1;
    if [ $FSIZE -gt 0 ]
    then
      exit -1;
    fi
}

TEST_DIR=$CMSSW_BASE/src/SimG4CMS/Forward/test/python

F1=${TEST_DIR}/runMTDSens_cfg.py
F2=${TEST_DIR}/runMTDSens_DD4hep_cfg.py

REF_FILE="Geometry/TestReference/data/mtdCommonDataRef.log.gz"
REF=""
for d in $(echo $CMSSW_SEARCH_PATH | tr ':' '\n') ; do
  if [ -e "${d}/${REF_FILE}" ] ; then
    REF="${d}/${REF_FILE}"
      break
  fi
done
[ -z $REF ] && exit 1

zcat $REF > ./mtdCommonDataRef.log
sort -n ./mtdCommonDataRef.log > ./tmplog; mv ./tmplog ./mtdCommonDataRef.log
gzip ./mtdCommonDataRef.log
REF=${PWD}/mtdCommonDataRef.log.gz

FILE1=mtdG4sensDDD.log
FILE2=mtdG4sensDD4hep.log
LOG=mtdg4sdlog
DIF=mtdg4sdif

echo " testing SimG4CMS/Forward"

echo "===== Test \"cmsRun runMTDSens_cfg.py\" ===="
rm -f $LOG $DIF $FILE1

cmsRun $F1 >& $LOG || die "Failure using cmsRun $F1" $?
sort -n $FILE1 > tmpF1; mv tmpF1 $FILE1
gzip -f $FILE1 || die "$FILE1 compression fail" $?
(zdiff $FILE1.gz $REF >& $DIF || [ -s $DIF ] && checkDiff $DIF || echo "OK") || die "Failure in comparison for $FILE1" $?

rm -f $LOG $DIF $FILE2
echo "===== Test \"cmsRun runMTDSens_DD4hep_cfg.py\" ===="

cmsRun $F2 >& $LOG || die "Failure using cmsRun $F2" $?
sort -n $FILE2 > tmpF2; mv tmpF2 $FILE2
gzip -f $FILE2 || die "$FILE2 compression fail" $?
(zdiff $FILE2.gz $REF >& $DIF || [ -s $DIF ] && checkDiff $DIF || echo "OK") || die "Failure in comparison for $FILE2" $?

