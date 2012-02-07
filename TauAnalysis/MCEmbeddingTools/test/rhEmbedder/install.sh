#! /bin/bash

wd=`pwd`
targetDir=$CMSSW_BASE/src/
cp DetAssoc.patch $targetDir
cd $targetDir
addpkg TrackingTools/TrackAssociator
patch -p0 -i DetAssoc.patch

cvs co TauAnalysis/Skimming/python/goldenZmmSelectionVBTFrelPFIsolation_cfi.py

scramv1 b -j 12
