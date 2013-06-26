#! /bin/bash
export RELEASE=$CMSSW_VERSION

export refDir=/afs/cern.ch/cms/performance/tracker/activities/validation/ReferenceFiles/$RELEASE

if [ ! -d $refDir ]; then
    echo Creating directory $refDir
    mkdir $refDir
fi

export geomDir=Geometry

if [ ! -d $geomDir ]; then
    echo Creating directory $geomDir/.
    mkdir $geomDir
fi

# Copy Reference Root files
cp matbdg_BeamPipe.root      $geomDir/.
cp matbdg_PixBar.root        $geomDir/.
cp matbdg_PixFwdMinus.root   $geomDir/.
cp matbdg_PixFwdPlus.root    $geomDir/.
cp matbdg_TEC.root           $geomDir/.
cp matbdg_InnerServices.root $geomDir/.
cp matbdg_TIB.root           $geomDir/.
cp matbdg_TIDB.root          $geomDir/.
cp matbdg_TIDF.root          $geomDir/.
cp matbdg_TkStrct.root       $geomDir/.
cp matbdg_TOB.root           $geomDir/.
cp matbdg_Tracker.root       $geomDir/.

# Copy Reference Text Files
cp ModuleInfo.log      $geomDir/.
cp ModuleNumbering.dat $geomDir/.

# Copy overlap report
cp trackerOverlap.log $geomDir/.

# Copy Images
cp -R Images $geomDir/.

tar -cvzf $geomDir.tgz $geomDir

mv $geomDir.tgz $refDir/.

rm -r $geomDir
