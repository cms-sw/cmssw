#! /bin/bash
export RELEASE=$CMSSW_VERSION

export wwwDir=/afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/

#export wwwDir=./prova

if [ ! -d $wwwDir ]; then
    echo Creating directory $wwwDir
    mkdir $wwwDir
fi


export geomDir=$wwwDir/TrackerGeometry

if [ ! -d $geomDir ]; then
    echo Creating directory $geomDir
    mkdir $geomDir
fi

# Material Budget
export mbDir=$geomDir/MaterialBudget
# Geometry: Positioning and Numbering Scheme
export nsDir=$geomDir/Geometry

# make Material Budget directories
mkdir $mbDir
mkdir $mbDir/Comparison
mkdir $mbDir/Comparison/eps
#mkdir $mbDir/Comparison/gif
mkdir $mbDir/Plots
mkdir $mbDir/Plots/eps
mkdir $mbDir/Plots/gif

# make Geometry
mkdir $nsDir


# move Material Budget plots
mv Images/*Comparison*.eps $mbDir/Comparison/eps/.
#mv Images/*Comparison*.gif $mbDir/Comparison/gif/.
mv Images/*.eps            $mbDir/Plots/eps/.
mv Images/*.gif            $mbDir/Plots/gif/.

# move geometry diff files
mv diff_info.temp     $nsDir/ModulePositioning.diff
mv diff_num.temp      $nsDir/ModuleNumbering.diff
mv NumberingInfo.log  $nsDir/.
mv num.log            $nsDir/.
