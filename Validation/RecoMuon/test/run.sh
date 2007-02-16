#!/bin/bash 

#WORKDIR=/afs/cern.ch/user/j/jhgoh/Work/Validation/CMSSW_1_2_0/src/Validation/RecoMuon
#OUTDIR=$WORKDIR/data/`date %Y%m%d%I%M`

cmsRun $PKGDIR/test/${1}.cfg >& ${1}.log
RETVAL=$?

if [ $RETVAL != 0 ]; then
  tar czf ${1}.log.tgz ${1}.log
  mv -f ${1}.log.tgz  $OUTDIR/
fi

rm -f ${1}.log
mv -f *.root $OUTDIR/
