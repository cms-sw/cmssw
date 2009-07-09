#!/bin/csh 

#setenv RELDIR CMSSW_3_1_1/MC_noPU_ootb
setenv RELDIR CMSSW_3_1_1/STARTUP_noPU_ootb
setenv PKGDIR $CMSSW_BASE/src/Validation/RecoMuon/test
setenv OUTDIR $PKGDIR/RootHisto_`date +%Y%m%d%H%M`

[ -d $OUTDIR ] || mkdir $OUTDIR

#setenv QUEUE 1nd
#setenv QUEUE 8nh
setenv QUEUE 1nh
#setenv QUEUE 8nm

setenv RUNSCRIPT run.sh

#  setenv MYJOB1 RelValSingleMuPt10
#  bsub -q $QUEUE -oo $OUTDIR/$MYJOB1.out $RUNSCRIPT $MYJOB1
#  setenv MYJOB2 RelValSingleMuPt100
#  bsub -q $QUEUE -oo $OUTDIR/$MYJOB2.out $RUNSCRIPT $MYJOB2
#  setenv MYJOB3 RelValSingleMuPt1000
#  bsub -q $QUEUE -oo $OUTDIR/$MYJOB3.out $RUNSCRIPT $MYJOB3
  setenv MYJOB4 RelValTTbar
  bsub -q $QUEUE -oo $OUTDIR/$MYJOB4.out $RUNSCRIPT $MYJOB4
  setenv MYJOB5 RelValZMM
  bsub -q $QUEUE -oo $OUTDIR/$MYJOB5.out $RUNSCRIPT $MYJOB5
