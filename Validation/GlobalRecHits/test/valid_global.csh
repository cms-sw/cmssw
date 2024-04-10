#! /bin/csh
 
eval `scramv1 ru -csh`

setenv GLBLREFDIR /afs/cern.ch/cms/data/CMSSW/Validation/GlobalRecHitss/data
setenv LOCLREFDIR ${CMSSW_RELEASE_BASE}/src/Validation/GlobalRecHitss/data

echo "============> Validating Global RecHits <============"
echo "......producing output file with this release"
if ( -e MessageLogger.log ) rm MessageLogger.log
if ( -e output.log ) rm output.log
if ( -e FrameworkJobReport.xml ) rm FrameworkJobReport.xml
#comment out if RefPoolSource.cfi modified to use /afs location directly
#cp ${GLBLREFDIR}/MC_010p2_minbias.root .
cmsRun DetSim+Digi+Reco+Global.cfg >& output.log
echo "......creating histogram file with this release"
root -b -q MakeHistograms.C\(\"GlobalRecHits.root\",\"GlobalRecHitsHistograms\"\)
echo "......comparing against reference file from previous release"
cp ${LOCLREFDIR}/GlobalRecHitsHistograms-reference.root .
root -b -q MakeValidation.C\(\"GlobalRecHitsHistograms.root\",\"GlobalRecHitsHistograms-reference.root\",\"GlobalRecHitsHistogramsCompare\"\)
echo "......results of validation in GlobalRecHitsHistogramsCompare.ps"
echo "============> Validating Global RecHits <============"
