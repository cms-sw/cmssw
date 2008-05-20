#! /bin/csh

eval `scramv1 runtime -csh`

setenv DATADIR $CMSSW_BASE/src
setenv REFDIRS /afs/cern.ch/cms/performance/tracker/activities/validation/ReferenceFiles
setenv REFDIR $REFDIRS/${1}
setenv NEWREFDIR $REFDIRS/$CMSSW_VERSION
if ( ! -e $NEWREFDIR ) mkdir $NEWREFDIR
if ( ! -e $NEWREFDIR/SimHits ) mkdir $NEWREFDIR/SimHits
if ( ! -e $NEWREFDIR/Digis ) mkdir $NEWREFDIR/Digis
if ( ! -e $NEWREFDIR/RecHits ) mkdir $NEWREFDIR/RecHits
if ( ! -e $NEWREFDIR/LiteGeometry ) mkdir $NEWREFDIR/LiteGeometry
if ( ! -e $NEWREFDIR/TrackingRecHits ) mkdir $NEWREFDIR/TrackingRecHits
if ( ! -e $NEWREFDIR/Tracks ) mkdir $NEWREFDIR/Tracks


cd ${DATADIR}

project CMSSW

# Get the relevant packages
#
cvs co -r $CMSSW_VERSION Validation/Geometry
cvs co -r $CMSSW_VERSION Validation/TrackerHits
cvs co -r $CMSSW_VERSION Validation/TrackerDigis
cvs co -r $CMSSW_VERSION Validation/TrackerRecHits
cvs co -r $CMSSW_VERSION Validation/TrackingMCTruth
cvs co -r $CMSSW_VERSION Validation/RecoTrack
#Add also co of CalibTracker for test o Lite version of Geometry
cvs co -r $CMSSW_VERSION CalibTracker/SiStripCommon
#
# Geometry Validation
#
if ('${2}' == 'GEOMETRY' ) then
cd ${DATADIR}/Validation/Geometry/test
./TrackerGeometryValidation.sh ${1}
chmod a+x copyWWWTrackerGeometry.sh    
./copyWWWTrackerGeometry.sh
endif
#
# Run validation chain
#
cd ${DATADIR}/Validation/TrackerConfiguration/test

#cp /afs/cern.ch/cms/data/CMSSW/Validation/TrackerHits/data/Muon.root .

#cmsRun Muon_FullChain.cfg

cmsRun ValidationChainOnly.cfg
#if ( -e Muon.root ) /bin/rm Muon.root

cp ./TrackerHitHisto.root ${DATADIR}/Validation/TrackerHits/test/

cd ${DATADIR}/Validation/TrackerHits/test 

cp ${REFDIR}/SimHits/* ../data/

cp TrackerHitHisto.root $NEWREFDIR/SimHits

if ( ! -e plots ) mkdir plots
root -b -p -q SiStripHitsCompareEnergy.C
if ( ! -e plots/muon ) mkdir plots/muon
gzip *.eps
mv eloss*.eps.gz plots/muon
mv eloss*.gif plots/muon

root -b -p -q SiStripHitsComparePosition.C
if ( ! -e plots/muon ) mkdir plots/muon
gzip *.eps
mv pos*.eps.gz plots/muon
mv pos*.gif plots/muon

source copyWWWall.csh

cd ${DATADIR}/Validation/TrackerDigis/test 

cp ${DATADIR}/Validation/TrackerConfiguration/test/stripdigihisto.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/pixeldigihisto.root .

cp ${REFDIR}/Digis/* ../data/

cp pixeldigihisto.root $NEWREFDIR/Digis
cp stripdigihisto.root $NEWREFDIR/Digis

root -b -p -q  SiPixelDigiCompare.C
gzip *.eps
source copyWWWPixel.csh
root -b -p -q  SiStripDigiCompare.C
gzip *.eps
source copyWWWStrip.csh

cd ${DATADIR}/Validation/TrackerRecHits/test

cp ${DATADIR}/Validation/TrackerConfiguration/test/sistriprechitshisto.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/pixelrechitshisto.root .

cp ${REFDIR}/RecHits/* ../data/

cp sistriprechitshisto.root $NEWREFDIR/RecHits/
cp pixelrechitshisto.root $NEWREFDIR/RecHits/

root -b -p -q SiPixelRecHitsCompare.C
gzip *.eps
source copyWWWPixel.csh
root -b -p -q SiStripRecHitsCompare.C
gzip *.eps
source copyWWWStrip.csh

cd ${DATADIR}/Validation/TrackingMCTruth/test
cp ${DATADIR}/Validation/TrackerConfiguration/test/trackingtruthhisto.root .
cp ${REFDIR}/TrackingParticles/* ../data/
cp trackingtruthhisto.root $NEWREFDIR/TrackingParticles/


root -b -p -q TrackingTruthCompare.C
gzip *.eps
source copyWWWTP.csh

cd ${DATADIR}/Validation/RecoTrack/test

cp ${DATADIR}/Validation/TrackerConfiguration/test/validationPlots.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/pixeltrackingrechitshist.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/striptrackingrechitshisto.root .

cp ${REFDIR}/Tracks/* ../data/
cp ${REFDIR}/TrackingRecHits/* ../data/

cp validationPlots.root $NEWREFDIR/Tracks/
cp pixeltrackingrechitshist.root $NEWREFDIR/TrackingRecHits/
cp striptrackingrechitshisto.root $NEWREFDIR/TrackingRecHits/

root -b -p -q TracksCompareChain.C
gzip *.eps
source copyWWWTracks.csh

root -b -p -q SiPixelRecoCompare.C 
gzip *.eps
source copyWWWPixel.csh

root -b -p -q SiStripTrackingRecHitsCompare.C 
gzip *.eps
source copyWWWStrip.csh

#Check on the fly in order to check the correctness of LiteGeometry
cd ${DATADIR}/CalibTracker/SiStripCommon/test
cp ${REFDIR}/LiteGeometry/* oldgeometrylite.txt
cmsRun writeFile.cfg
cp myfile.txt $NEWREFDIR/LiteGeometry/

diff myfile.txt oldgeometrylite.txt > ! litegeometryDIFF.txt
mkdir /afs/cern.ch/cms/performance/tracker/activities/validation/${1}/LiteGeometry
cp litegeometryDIFF.txt  /afs/cern.ch/cms/performance/tracker/activities/validation/${1}/LiteGeometry
