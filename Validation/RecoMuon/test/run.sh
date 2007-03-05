#!/bin/bash 

cmsRun $PKGDIR/test/${1}.cfg >& ${1}.log
RETVAL=$?

if [ $RETVAL != 0 ]; then
  tar czf ${1}.log.tgz ${1}.log
  mv -f ${1}.log.tgz  $OUTDIR/
fi

rm -f ${1}.log
mv -f *.root $OUTDIR/
