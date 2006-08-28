#! /bin/csh
 
eval `scramv1 ru -csh`

setenv GLBLREFDIR /afs/cern.ch/cms/data/CMSSW/Validation/GlobalHits/data

echo "============> Validating Global Sim Hits <============"
echo "......producing output file with this release"
if ( -e MessageLogger.log ) rm MessageLogger.log
if ( -e output.log ) rm output.log
if ( -e FrameworkJobReport.xml ) rm FrameworkJobReport.xml
#comment out the cp if RefPoolSource.cfi has been modified to use /afs location directly
#cp ${GLBLREFDIR}/MC_090p3_minbias.root .
cmsRun -p DetSim+Global.cfg >& output.log
echo "......comparing against reference file from previous release"
cp ${CMSSW_RELEASE_BASE}/src/Validation/GlobalHits/data/GlobalHitsHistograms-reference.root .
root -b -q MakeValidation.C
echo "......results of validation in GlobalHitsHistogramsCompare.ps"
echo "============> Validating Global Sim Hits <============"
