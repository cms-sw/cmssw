#! /bin/csh
 
eval `scramv1 ru -csh`

setenv GLBLREFDIR /afs/cern.ch/cms/data/CMSSW/Validation/GlobalHits/data
#setenv GLBLREFDIR /uscms/home/strang/simval/CMSSW_0_6_1/src/Validation/GlobalHits/data

echo "============> Validating Global Sim Hits <============"
echo "......producing output file with this release"
if ( -e MessageLogger.log ) rm MessageLogger.log
if ( -e output.log ) rm output.log
if ( -e FrameworkJobReport.xml ) rm FrameworkJobReport.xml
#cp $GLBLREFDIR/MC_060_minbias.root .
cmsRun -p DetSim+Global.cfg >& output.log
echo "......comparing against reference file from previous release"
cp $GLBLREFDIR/GlobalValHistograms-reference.root .
root -b -q MakeValidation.C
echo "......results of validation in GlobalValHistogramsCompare.ps"
echo "============> Validating Global Sim Hits <============"
