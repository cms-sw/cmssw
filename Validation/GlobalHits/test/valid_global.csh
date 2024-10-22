#! /bin/csh
 
eval `scramv1 ru -csh`

setenv GLBLREFDIR /afs/cern.ch/cms/data/CMSSW/Validation/GlobalHits/data
setenv LOCLREFDIR ${CMSSW_RELEASE_BASE}/src/Validation/GlobalHits/data

echo "============> Validating Global Sim Hits <============"
echo "......producing output file with this release"
if ( -e MessageLogger.log ) rm MessageLogger.log
if ( -e output.log ) rm output.log
if ( -e FrameworkJobReport.xml ) rm FrameworkJobReport.xml
#comment out if RefPoolSource.cfi modified to use /afs location directly
#cp ${GLBLREFDIR}/MC_010p2_minbias.root .
cmsRun DetSim+Global.cfg >& output.log
echo "......creating histogram file with this release"
root -b -q MakeHistograms.C\(\"GlobalHits.root\",\"GlobalHitsHistograms\"\)
echo "......comparing against reference file from previous release"
cp ${LOCLREFDIR}/GlobalHitsHistograms-reference.root .
root -b -q MakeValidation.C\(\"GlobalHitsHistograms.root\",\"GlobalHitsHistograms-reference.root\",\"GlobalHitsHistogramsCompare\"\)
echo "......results of validation in GlobalHitsHistogramsCompare.ps"
echo "============> Validating Global Sim Hits <============"
