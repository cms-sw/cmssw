#!/bin/csh 

setenv PKGDIR $CMSSW_BASE/src/Validation/RecoMuon/test
setenv OUTDIR $PKGDIR/RootHisto_`date +%Y%m%d%H%M`

[ -d $OUTDIR ] || mkdir $OUTDIR

#setenv QUEUE 1nd
setenv QUEUE 8nh
#setenv QUEUE 1nh
#setenv QUEUE 8nm

setenv RUNSCRIPT run.sh

  setenv MYJOB1 RelValSingleMuPt100
  bsub -q $QUEUE -oo $OUTDIR/$MYJOB1.out $RUNSCRIPT $MYJOB1
  setenv MYJOB2 RelValSingleMuPt1000
  bsub -q $QUEUE -oo $OUTDIR/$MYJOB2.out $RUNSCRIPT $MYJOB2
