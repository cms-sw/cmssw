#! /bin/csh
 
eval `scramv1 ru -csh`

setenv GLBLREFDIR /afs/cern.ch/cms/data/CMSSW/Validation/GlobalDigis/data
setenv LOCLREFDIR ${CMSSW_RELEASE_BASE}/src/Validation/GlobalDigis/data

echo "============> Validating Global Digi Hits <============"
echo "......producing output file with this release"
if ( -e MessageLogger.log ) rm MessageLogger.log
if ( -e output.log ) rm output.log
if ( -e FrameworkJobReport.xml ) rm FrameworkJobReport.xml
#comment out if RefPoolSource.cfi modified to use /afs location directly
#cp ${GLBLREFDIR}/MC_010p2_minbias.root .
cmsRun DetSim+Digi+Global.cfg >& output.log
echo "......creating histogram file with this release"
root -b -q MakeHistograms.C\(\"GlobalDigis.root\",\"GlobalDigisHistograms\"\)
echo "......comparing against reference file from previous release"
cp ${LOCLREFDIR}/GlobalDigisHistograms-reference.root .
root -b -q MakeValidation.C\(\"GlobalDigisHistograms.root\",\"GlobalDigisHistograms-reference.root\",\"GlobalDigisHistogramsCompare\"\)
echo "......results of validation in GlobalDigisHistogramsCompare.ps"
echo "============> Validating Global Digi Hits <============"
