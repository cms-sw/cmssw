#!/bin/csh 

setenv RELEASE CMSSW_3_2_3

setenv PKGDIR $CMSSW_BASE/src/Validation/RecoMuon/test


setenv QUEUE 1nw
#setenv QUEUE 1nd
#setenv QUEUE 8nh
#setenv QUEUE 1nh
#setenv QUEUE 8nm

setenv RUNSCRIPT run.sh

setenv OUTDIR $PKGDIR/RootHistoMC_`date +%Y%m%d%H%M`
[ -d $OUTDIR ] || mkdir $OUTDIR
setenv RELDIR $RELEASE/MC_noPU_ootb
  setenv MYJOB1 RelValSingleMuPt10
  bsub -q $QUEUE -oo $OUTDIR/$MYJOB1.out $RUNSCRIPT $MYJOB1
  setenv MYJOB2 RelValSingleMuPt100
  bsub -q $QUEUE -oo $OUTDIR/$MYJOB2.out $RUNSCRIPT $MYJOB2
  setenv MYJOB3 RelValSingleMuPt1000
  bsub -q $QUEUE -oo $OUTDIR/$MYJOB3.out $RUNSCRIPT $MYJOB3
  setenv MYJOB4 RelValTTbar
  bsub -q $QUEUE -oo $OUTDIR/$MYJOB4.out $RUNSCRIPT $MYJOB4

setenv OUTDIR $PKGDIR/RootHistoSTARTUP_`date +%Y%m%d%H%M`
[ -d $OUTDIR ] || mkdir $OUTDIR
setenv RELDIR $RELEASE/STARTUP_noPU_ootb
  setenv MYJOB5 RelValTTbar
  bsub -q $QUEUE -oo $OUTDIR/$MYJOB5.out $RUNSCRIPT $MYJOB5
  setenv MYJOB6 RelValZMM
  bsub -q $QUEUE -oo $OUTDIR/$MYJOB6.out $RUNSCRIPT $MYJOB6
  setenv MYJOB7 RelValCosmics
  bsub -q $QUEUE -oo $OUTDIR/$MYJOB7.out $RUNSCRIPT $MYJOB7
