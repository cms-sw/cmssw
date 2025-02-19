#! /bin/csh

eval `scramv1 runtime -csh`

setenv DATADIR $CMSSW_BASE/src

cd ${DATADIR}

project CMSSW

cvs co Validation/TrackerHits
cvs co Validation/TrackerDigis


cd ${DATADIR}/Validation/TrackerRecHits/test

cp /afs/cern.ch/cms/data/CMSSW/Validation/TrackerHits/data/Muon.root .

cmsRun Muon_FullChain.cfg

if ( -e Muon.root ) /bin/rm Muon.root

cp ./Muon_FullValidation.root ${DATADIR}/Validation/TrackerHits/test/SimHitMuon.root

cd ${DATADIR}/Validation/TrackerHits/test 

source Compare_muon_all.csh

source copyWWWall.csh

cd ${DATADIR}/Validation/TrackerDigis/test 

cp ${DATADIR}/Validation/TrackerRecHits/test/stripdigihisto.root .
cp ${DATADIR}/Validation/TrackerRecHits/test/pixeldigihisto.root .

root -b -p -q  SiPixelDigiCompare.C
source copyWWWPixel.csh
root -b -p -q  SiStripDigiCompare.C
source copyWWWStrip.csh

cd ${DATADIR}/Validation/TrackerRecHits/test

root -b -p -q SiPixelRecHitsCompare.C
source copyWWWPixel.csh
root -b -p -q SiStripRecHitsCompare.C
source copyWWWStrip.csh

