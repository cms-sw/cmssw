#! /bin/csh

eval `scramv1 runtime -csh`

setenv DATADIR $CMSSW_BASE/src

cd ${DATADIR}

project CMSSW
#
# Get the relevant packages
#
cvs co Validation/Geometry
cvs co Validation/TrackerHits
cvs co Validation/TrackerDigis
cvs co Validation/TrackerRecHits
cvs co Validation/RecoTrack
#
# Geometry Validation
#
cd ${DATADIR}/Validation/Geometry/test
./TrackerGeometryValidation.sh
./copyWWWTrackerGeometry.sh
#
# Run validation chain
#
cd ${DATADIR}/Validation/TrackerConfiguration/test

cp /afs/cern.ch/cms/data/CMSSW/Validation/TrackerHits/data/Muon.root .

cmsRun Muon_FullChain.cfg

if ( -e Muon.root ) /bin/rm Muon.root

cp ./Muon_FullValidation.root ${DATADIR}/Validation/TrackerHits/test/SimHitMuon.root

cd ${DATADIR}/Validation/TrackerHits/test 

source Compare_muon_all.csh

source copyWWWall.csh

cd ${DATADIR}/Validation/TrackerDigis/test 

cp ${DATADIR}/Validation/TrackerConfiguration/test/stripdigihisto.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/pixeldigihisto.root .

root -b -p -q  SiPixelDigiCompare.C
source copyWWWPixel.csh
root -b -p -q  SiStripDigiCompare.C
source copyWWWStrip.csh

cd ${DATADIR}/Validation/TrackerRecHits/test

cp ${DATADIR}/Validation/TrackerConfiguration/test/sistriprechitshisto.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/pixelrechitshisto.root .

root -b -p -q SiPixelRecHitsCompare.C
source copyWWWPixel.csh
root -b -p -q SiStripRecHitsCompare.C
source copyWWWStrip.csh

cd ${DATADIR}/Validation/RecoTrack/test

cp ${DATADIR}/Validation/TrackerConfiguration/test/validationPlots.root .

root -b -p -q TracksCompare.C
source copyWWWTracks.csh


