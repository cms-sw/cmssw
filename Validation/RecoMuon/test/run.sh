#!/bin/bash 

cd $CMSSW_BASE/src
eval `scramv1 runtime -sh`

cd $WORKDIR
echo " pwd ->"
pwd


cmsRun $PKGDIR/$RELDIR/${1}/${1}_1.py >& ${1}_1.log
RETVAL=$?

#if [ $RETVAL != 0 ]; then
#  tar czf ${1}.log.tgz ${1}.log
#  mv -f *.tgz  $OUTDIR/
#fi

#mv  DQM_V0001_R000000001__MC_31X_V1__${1}__Validation.root val.${1}.root
mv  DQM_V0001_R000000001__STARTUP31X_V1__${1}__Validation.root val.${1}.root

mv -f ${1}_1.log $OUTDIR/
mv -f *.root $OUTDIR/
