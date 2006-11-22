#! /bin/bash

export here=$PWD
cd $here
echo "Working area:" $here
eval `scramv1 runtime -sh`

export referenceDir=/afs/cern.ch/cms/data/CMSSW/Validation/Geometry/reference/Tracker
echo "Reference area:" $referenceDir

# Create Images/ directory if it does not exist
if [ ! -d Images ]; then
    echo "Creating directory Images/"
    mkdir Images
fi
#

# Download the source file
echo "Download the Monte Carlo source file..."
wget `cat $CMSSW_RELEASE_BASE/src/Validation/Geometry/data/download.url`
echo "...done"
#

# Download the reference files and rename them to 'old'
echo "Download the reference 'old' files..."
cp $referenceDir/matbdg_TkStrct.root     matbdg_TkStrct_old.root 
cp $referenceDir/matbdg_PixBar.root      matbdg_PixBar_old.root 
cp $referenceDir/matbdg_PixFwdPlus.root  matbdg_PixFwdPlus_old.root 
cp $referenceDir/matbdg_PixFwdMinus.root matbdg_PixFwdMinus_old.root 
cp $referenceDir/matbdg_TIB.root         matbdg_TIB_old.root 
cp $referenceDir/matbdg_TIDF.root        matbdg_TIDF_old.root 
cp $referenceDir/matbdg_TIDB.root        matbdg_TIDB_old.root 
cp $referenceDir/matbdg_TOB.root         matbdg_TOB_old.root 
cp $referenceDir/matbdg_TEC.root         matbdg_TEC_old.root 
cp $referenceDir/matbdg_Tracker.root     matbdg_Tracker_old.root 
cp $referenceDir/matbdg_BeamPipe.root    matbdg_BeamPipe_old.root 
echo "...done"
#

# Run all the Tracker scripts and rename files as 'new'
echo "Run all the scripts to produce the 'new' files..."
cmsRun runP_TkStrct.cfg     > TkStrct.txt
cmsRun runP_PixBar.cfg      > PixBar.txt
cmsRun runP_PixFwdPlus.cfg  > PixFwdPlus.txt
cmsRun runP_PixFwdMinus.cfg > PixFwdMinus.txt
cmsRun runP_TIB.cfg         > TIB.txt
cmsRun runP_TIDF.cfg        > TIDF.txt
cmsRun runP_TIDB.cfg        > TIDB.txt
cmsRun runP_TOB.cfg         > TOB.txt
cmsRun runP_TEC.cfg         > TEC.txt
cmsRun runP_Tracker.cfg     > Tracker.txt
cmsRun runP_BeamPipe.cfg    > BeamPipe.txt
#
cp matbdg_TkStrct.root     matbdg_TkStrct_new.root 
cp matbdg_PixBar.root      matbdg_PixBar_new.root 
cp matbdg_PixFwdPlus.root  matbdg_PixFwdPlus_new.root 
cp matbdg_PixFwdMinus.root matbdg_PixFwdMinus_new.root 
cp matbdg_TIB.root         matbdg_TIB_new.root 
cp matbdg_TIDF.root        matbdg_TIDF_new.root 
cp matbdg_TIDB.root        matbdg_TIDB_new.root 
cp matbdg_TOB.root         matbdg_TOB_new.root 
cp matbdg_TEC.root         matbdg_TEC_new.root 
cp matbdg_Tracker.root     matbdg_Tracker_new.root 
cp matbdg_BeamPipe.root    matbdg_BeamPipe_new.root 
echo "...done"
#

# Produce the 'new' plots
echo "Run the Tracker macro MaterialBudget.C to produce the 'new' plots..."
root -b -q 'MaterialBudget.C("PixBar")'
root -b -q 'MaterialBudget.C("PixFwdPlus")'
root -b -q 'MaterialBudget.C("PixFwdMinus")'
root -b -q 'MaterialBudget.C("TIB")'
root -b -q 'MaterialBudget.C("TIDF")'
root -b -q 'MaterialBudget.C("TIDB")'
root -b -q 'MaterialBudget.C("TOB")'
root -b -q 'MaterialBudget.C("TEC")'
root -b -q 'MaterialBudget.C("TkStrct")'
root -b -q 'MaterialBudget.C("Tracker")'
root -b -q 'MaterialBudget.C("TrackerSum")'
root -b -q 'MaterialBudget.C("Pixel")'
root -b -q 'MaterialBudget.C("Strip")'
root -b -q 'MaterialBudget_TDR.C()'
echo "...done"
#

# Compare 'old' and 'new' plots
echo "Run the Tracker macro TrackerMaterialBudgetComparison.C to compare 'old and 'new' plots..."
root -b -q 'TrackerMaterialBudgetComparison.C("PixBar")'
root -b -q 'TrackerMaterialBudgetComparison.C("PixFwdPlus")'
root -b -q 'TrackerMaterialBudgetComparison.C("PixFwdMinus")'
root -b -q 'TrackerMaterialBudgetComparison.C("TIB")'
root -b -q 'TrackerMaterialBudgetComparison.C("TIDF")'
root -b -q 'TrackerMaterialBudgetComparison.C("TIDB")'
root -b -q 'TrackerMaterialBudgetComparison.C("TOB")'
root -b -q 'TrackerMaterialBudgetComparison.C("TEC")'
root -b -q 'TrackerMaterialBudgetComparison.C("TkStrct")'
root -b -q 'TrackerMaterialBudgetComparison.C("Tracker")'
root -b -q 'TrackerMaterialBudgetComparison.C("TrackerSum")'
root -b -q 'TrackerMaterialBudgetComparison.C("Pixel")'
root -b -q 'TrackerMaterialBudgetComparison.C("Strip")'
echo "...done"
#

# Run the Tracker ModuleInfo analyzer (to compare position/orientation of Tracker Modules)
echo "Run the Tracker ModuleInfo analyzer to print Tracker Module info (position/orientation)..."
cmsRun $CMSSW_RELEASE_BASE/src/Geometry/TrackerGeometryBuilder/test/trackerModuleInfo.cfg
echo "...done"
#

# Compare the ModuleInfo.log file with the reference one
echo "Compare the ModuleInfo.log (Tracker Module position/orientation) file with the reference one..."
if [ -e diff.temp ]; then
    rm -rf diff.temp
fi
#
diff ModuleInfo.log $referenceDir/ModuleInfo.log > diff.temp
if [ -s diff.temp ]; then
    echo "WARNING: the module position/orientation is changed, check diff.temp file for details"
else
    echo "Tracker Module position/orientation OK"
fi
echo "...done"
#

echo "TRACKER GEOMETRY VALIDATION ENDED... LOOK AT THE RESULTS"
