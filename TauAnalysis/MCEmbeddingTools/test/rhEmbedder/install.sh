#! /bin/bash

wd=`pwd`
targetDir=$CMSSW_BASE/src/
cp DetAssoc.patch $targetDir
cd $targetDir
addpkg TrackingTools/TrackAssociator
patch -p0 -i DetAssoc.patch


