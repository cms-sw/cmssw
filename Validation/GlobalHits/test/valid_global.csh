#! /bin/csh
 
eval `scramv1 ru -csh`

echo "============> Validating Global Sim Hits <============"
echo "......producing output file with this release"
rm MessageLogger.log
rm output.log
rm FrameworkJobReport.xml
cmsRun -p EvtGen+DetSim+Global.cfg >& output.log
echo "......comparing against reference file from previous release"
root -b -q MakeValidation.C
echo "......results of validation in GlobalValHistogramsCompare.ps"
echo "......copying current output to reference_new"
cp GlobalValHistograms.ps GlobalValHistograms-reference_new.ps
cp GlobalValHistograms.root GlobalValHistograms-reference_new.root
cp GlobalValProducer.root GlobalValProducer-reference_new.root
echo "============> Validating Global Sim Hits <============"
