#! /bin/bash
export RELEASE=$CMSSW_VERSION

export wwwDir=/afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE

if [ ! -d $wwwDir ]; then
    echo Creating directory $wwwDir
    mkdir $wwwDir
fi


export geomDir=$wwwDir/TrackerGeometry

if [ ! -d $geomDir ]; then
    echo Creating directory $geomDir
    mkdir $geomDir
fi

# Copy logfile
cp TrackerGeometryValidation.log $geomDir/.
#

# Material Budget
export mbDir=$geomDir/MaterialBudget
# Geometry: Positioning, Numbering Scheme and Overlaps
export nsDir=$geomDir/Geometry

# make Material Budget directories
mkdir $mbDir
mkdir $mbDir/Comparison
mkdir $mbDir/Comparison/eps
mkdir $mbDir/Comparison/gif
#mkdir $mbDir/Comparison/pdf
mkdir $mbDir/Plots
mkdir $mbDir/Plots/eps
mkdir $mbDir/Plots/gif
#mkdir $mbDir/Plots/pdf

# make Geometry
mkdir $nsDir


# move Material Budget plots
cp Images/*Comparison*.eps $mbDir/Comparison/eps/.
cp Images/*Comparison*.gif $mbDir/Comparison/gif/.
#cp Images/*Comparison*.pdf $mbDir/Comparison/pdf/.
for i in $(ls Images/*.eps | grep -v Comparison)
  do
  cp $i $mbDir/Plots/eps/.
done
for i in $(ls Images/*.gif | grep -v Comparison)
  do
  cp $i $mbDir/Plots/gif/.
done
#for i in $(ls Images/*.pdf | grep -v Comparison)
#  do
#  cp $i $mbDir/Plots/pdf/.
#done

tar -cvzf EPSFiles.tgz $mbDir/Comparison/eps
tar -cvzf GIFFiles.tgz $mbDir/Comparison/gif
rm $mbDir/Comparison/eps/*.eps
rm $mbDir/Comparison/gif/*.gif
mv EPSFiles.tgz $mbDir/Comparison/eps/.
mv GIFFiles.tgz $mbDir/Comparison/gif/.

tar -cvzf EPSFiles.tgz $mbDir/Plots/eps
tar -cvzf GIFFiles.tgz $mbDir/Plots/gif
rm $mbDir/Plots/eps/*.eps
rm $mbDir/Plots/gif/*.gif
mv EPSFiles.tgz $mbDir/Plots/eps/.
mv GIFFiles.tgz $mbDir/Plots/gif/.

# move geometry diff files
cp diff_info.temp     $nsDir/ModulePositioning.diff
cp diff_num.temp      $nsDir/ModuleNumbering.diff
cp NumberingInfo.log  $nsDir/.
cp num.log            $nsDir/.
cp trackerOverlap.log $nsDir/.

