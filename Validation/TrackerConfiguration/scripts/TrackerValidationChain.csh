#! /bin/csh

eval `scramv1 runtime -csh`

setenv DATADIR $CMSSW_BASE/src

setenv REFDIR /afs/cern.ch/cms/performance/tracker/activities/validation/ReferenceFiles/${1}

cd ${DATADIR}

project CMSSW
#
# Get the relevant packages
#
cvs co -r $CMSSW_VERSION Validation/Geometry
cvs co -r $CMSSW_VERSION Validation/TrackerHits
cvs co -r $CMSSW_VERSION Validation/TrackerDigis
cvs co -r $CMSSW_VERSION Validation/TrackerRecHits
cvs co -r $CMSSW_VERSION Validation/RecoTrack
#
# Geometry Validation
#
cd ${DATADIR}/Validation/Geometry/test
./TrackerGeometryValidation.sh ${1}
chmod a+x copyWWWTrackerGeometry.sh    
./copyWWWTrackerGeometry.sh
#
# Run validation chain
#
cd ${DATADIR}/Validation/TrackerConfiguration/test

cp /afs/cern.ch/cms/data/CMSSW/Validation/TrackerHits/data/Muon.root .

cmsRun Muon_FullChain.cfg

if ( -e Muon.root ) /bin/rm Muon.root

cp ./TrackerHitHisto.root ${DATADIR}/Validation/TrackerHits/test/

cd ${DATADIR}/Validation/TrackerHits/test 

cp ${REFDIR}/SimHits/* ../data/

if ( ! -e plots ) mkdir plots
root -b -p -q SiStripHitsCompareEnergy.C
if ( ! -e plots/muon ) mkdir plots/muon
mv eloss*.eps plots/muon
mv eloss*.gif plots/muon

root -b -p -q SiStripHitsComparePosition.C
if ( ! -e plots/muon ) mkdir plots/muon
mv pos*.eps plots/muon
mv pos*.gif plots/muon

source copyWWWall.csh

cd ${DATADIR}/Validation/TrackerDigis/test 

cp ${DATADIR}/Validation/TrackerConfiguration/test/stripdigihisto.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/pixeldigihisto.root .

cp ${REFDIR}/Digis/* ../data/

root -b -p -q  SiPixelDigiCompare.C
source copyWWWPixel.csh
root -b -p -q  SiStripDigiCompare.C
source copyWWWStrip.csh

cd ${DATADIR}/Validation/TrackerRecHits/test

cp ${DATADIR}/Validation/TrackerConfiguration/test/sistriprechitshisto.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/pixelrechitshisto.root .

cp ${REFDIR}/RecHits/* ../data/

root -b -p -q SiPixelRecHitsCompare.C
source copyWWWPixel.csh
root -b -p -q SiStripRecHitsCompare.C
source copyWWWStrip.csh

cd ${DATADIR}/Validation/RecoTrack/test

cp ${DATADIR}/Validation/TrackerConfiguration/test/validationPlots.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/pixeltrackingrechitshist.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/striptrackingrechitshisto.root .

cp ${REFDIR}/Tracks/* ../data/
cp ${REFDIR}/TrackingRecHits/* ../data/

root -b -p -q TracksCompareChain.C
source copyWWWTracks.csh

root -b -p -q SiPixelRecoCompare.C 
source copyWWWPixel.csh

root -b -p -q SiStripTrackingRecHitsCompare.C 
source copyWWWStrip.csh
