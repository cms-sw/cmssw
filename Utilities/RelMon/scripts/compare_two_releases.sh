#! /bin/bash 

# commands to run a comparison

RELEASE1=$1
RELEASE2=$2

echo About to compare $RELEASE1 and $RELEASE2

#-------------------------------------------------------------------------------

# Set Some useful variables ----------------------------------------------------
echo "Set Some useful variables..."

# The output directory name
COMPDIR="$RELEASE1"VS"$RELEASE2"

# The base directory on AFS
RELMONAFSBASE=/afs/cern.ch/cms/offline/dqm/ReleaseMonitoring
RELMONAFS="$RELMONAFSBASE"/"$COMPDIR"

# The base directory on Castor
RELMONCASTOR=/castor/cern.ch/user/d/dpiparo/TestRelMonOnCastor/"$COMPDIR"

# The number of simultaneous processes
NPROCESSES=6

# Fetch Files and Organise them ------------------------------------------------

# Full and FastSim: Get them from the GUI
echo "Fetching MC datasets..."
fetchall_from_DQM.py  $RELEASE1 -mc --p2 "START" 2>&1 |tee step_1_fetch_MC_rel1.log
fetchall_from_DQM.py  $RELEASE2 -mc --p2 "START" 2>&1 |tee step_1_fetch_MC_rel2.log

# Make directories and copy into them
mkdir FastSim
mv *FastSim*root FastSim
mkdir FullSim
mv *root FullSim

# Arrange files for a FullSimFastSim comparison
# create and enter the directory
FULLSIMFASTISMDIR=FullSimFastSim;
mkdir $FULLSIMFASTISMDIR;
cd  $FULLSIMFASTISMDIR;
# link all fastsim files
for FASTSIMFILE in `ls ../FastSim|grep "$RELEASE1"`; do 
  ln -s ../FastSim/"$FASTSIMFILE" .;
  done

# Link only those files that correspond to the FSim ones
# Isolate the dataset name
for DSET in `ls ../FastSim|sed 's/__/ /g'| cut -f2 -d " "`;do
  # The datasets can be more than one: e.g. pt10 or pt100
  FULLSIMFILES=`echo ../FullSim/*"$DSET"*"$RELEASE1"*`;
  # therefore loop on them
  for FULLSIMFILE in `echo $FULLSIMFILES`; do
    if [ -f $FULLSIMFILE ]; then
      ln -s $FULLSIMFILE .;
      fi;
    done; # end loop on fullsim datasets files matching the particular fastsim dataset
  done; # end loop on datasets
 get out of the dir
cd -

# Data: Get them from the GUI
echo "Fetching Data datasets..."
fetchall_from_DQM.py  $RELEASE1 -data 2>&1 |tee step_2_fetch_DATA_rel1.log
fetchall_from_DQM.py  $RELEASE2 -data 2>&1 |tee step_2_fetch_DATA_rel2.log

# Make directories and copy into them
mkdir Data
mv *root Data

# Creating dir on AFS -----------------------------------------------------------
echo "Creating directory on AFS"
mkdir $RELMONAFS

# Run the Comparisons, make the reports and copy them----------------------------
echo "Creating Reports"

echo " @@@ FastSim"
ValidationMatrix.py -a FastSim -o FastSimReport -N $NPROCESSES 2>&1 |tee step_3_reports_FastSim.log
echo "Compressing report for web"
dir2webdir.py FastSimReport 2>&1 |tee step_4_compress_FastSim.log
echo "Copying report on the web"
cp -r FastSimReport $RELMONAFS

echo " @@@ FastSim HLT"
ValidationMatrix.py -a FastSim -o FastSimReport_HLT -N $NPROCESSES --HLT 2>&1 |tee step_3_reports_FastSim_HLT.log
echo "Compressing report for web"
dir2webdir.py FastSimReport_HLT 2>&1 |tee step_4_compress_FastSim_HLT.log
echo "Copying report on the web"
cp -r FastSimReport_HLT $RELMONAFS

echo " @@@ FullSim"
ValidationMatrix.py -a FullSim -o FullSimReport -N $NPROCESSES 2>&1 |tee step_3_reports_FullSim.log
echo "Compressing report for web"
dir2webdir.py FullSimReport 2>&1 |tee step_4_compress_FullSim.log
echo "Copying report on the web"
cp -r FullSimReport $RELMONAFS

echo " @@@ FullSim_HLT"
ValidationMatrix.py -a FullSim -o FullSimReport_HLT -N $NPROCESSES --HLT 2>&1 |tee step_3_reports_FullSim_HLT.log
echo "Compressing report for web"
dir2webdir.py FullSimReport_HLT 2>&1 |tee step_4_compress_FullSim_HLT.log
echo "Copying report on the web"
cp -r FullSimReport_HLT $RELMONAFS

echo " @@@ FullSimFastSim"
FULLSIMFASTSIMREPORTDIR="$RELEASE1"_FullSimFastSimReport
ValidationMatrix.py -a $FULLSIMFASTISMDIR -o $FULLSIMFASTSIMREPORTDIR -N $NPROCESSES 2>&1 |tee step_3_reports_FullSimFastSim.log
echo "Compressing report for web"
dir2webdir.py $FULLSIMFASTSIMREPORTDIR 2>&1 |tee step_4_compress_FullSimFastSim.log
echo "Copying report on the web"
cp -r $FULLSIMFASTSIMREPORTDIR $RELMONAFSBASE

echo " @@@ FullSimFastSim"
FULLSIMFASTSIMREPORTDIR_HLT="$RELEASE1"_FullSimFastSimReport_HLT
ValidationMatrix.py -a $FULLSIMFASTISMDIR -o $FULLSIMFASTSIMREPORTDIR_HLT -N $NPROCESSES --HLT 2>&1 |tee step_3_reports_FullSimFastSim.log
echo "Compressing report for web"
dir2webdir.py $FULLSIMFASTSIMREPORTDIR_HLT 2>&1 |tee step_4_compress_FullSimFastSim_HLT.log
echo "Copying report on the web"
cp -r $FULLSIMFASTSIMREPORTDIR_HLT $RELMONAFSBASE

echo " @@@ FullSimFastSim_HLT"
export FULLSIMFASTSIMREPORTDIR_HLT="$RELEASE1"_FullSimFastSimReport_HLT
ValidationMatrix.py -a $FULLSIMFASTSIMDIR -o $FULLSIMFASTSIMREPORTDIR_HLT -N $NPROCESSES --HLT 2>&1 |tee step_3_reports_FullSimFastSim_HLT.log
echo "Compressing report for web"
dir2webdir.py $FULLSIMFASTSIMREPORTDIR_HLT 2>&1 |tee step_4_compress_FullSimFastSim_HLT.log
echo "Copying report on the web"
cp -r $FULLSIMFASTSIMREPORTDIR_HLT $RELMONAFSBASE

echo " @@@ Data"
ValidationMatrix.py -a Data -o DataReport -N $NPROCESSES 2>&1 |tee step_3_reports_Data.log
echo "Compressing report for web"
dir2webdir.py DataReport 2>&1 |tee step_4_compress_Data.log
echo "Copying report on the web"
cp -r DataReport $RELMONAFS


# copy everything on castor ----------------------------------------------------
echo "Backup of the material"
BACKUPDIR="$COMPDIR"Reports_HLT
mkdir $BACKUPDIR
mv *Report* $BACKUPDIR
tar -cvf - $BACKUPDIR | gzip > "$BACKUPDIR".tar.gz
/usr/bin/rfmkdir $RELMONCASTOR
/usr/bin/rfcp "$BACKUPDIR".tar.gz $RELMONCASTOR


