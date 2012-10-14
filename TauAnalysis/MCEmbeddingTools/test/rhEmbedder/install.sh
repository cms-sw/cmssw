#! /bin/bash

wd=`pwd`
targetDir=$CMSSW_BASE/src/

cp DetAssoc.patch $targetDir
cp MuonRecoPatch.patch $targetDir

cd $targetDir
addpkg RecoMuon/GlobalTrackFinder
patch -p0 -i MuonRecoPatch.patch

addpkg TrackingTools/TrackAssociator
patch -p0 -i DetAssoc.patch


