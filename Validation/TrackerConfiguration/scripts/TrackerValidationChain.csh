#! /bin/csh

set RefRelease="CMSSW_2_1_8"

# Possible values are:
# Reconstruction  : preforms reconstruction+validation + histograms comparison
# Validation : validation + histograms comparison
# Harvesting : harvesting + histogram comparison
# Comparison :histogram comparison

set Mode="Harvesting"
#set Mode="Reconstruction"
#set Mode="Validation"


# do you want to copy histograms on the validation page?

set copyWWW="false"
#set copyWWW="true"

# set the histogram file name in Comparison mode
set histogramfile=""

eval `scramv1 runtime -csh`

setenv DATADIR $CMSSW_BASE/src
setenv REFDIRS /afs/cern.ch/cms/performance/tracker/activities/validation/ReferenceFiles
setenv REFDIR $REFDIRS/$RefRelease
setenv NEWREFDIR $REFDIRS/$CMSSW_VERSION
if ( ! -e $NEWREFDIR ) mkdir $NEWREFDIR

#if ( ! -e $NEWREFDIR/SimHits ) mkdir $NEWREFDIR/SimHits
#if ( ! -e $NEWREFDIR/Digis ) mkdir $NEWREFDIR/Digis
#if ( ! -e $NEWREFDIR/RecHits ) mkdir $NEWREFDIR/RecHits
#if ( ! -e $NEWREFDIR/LiteGeometry ) mkdir $NEWREFDIR/LiteGeometry
#if ( ! -e $NEWREFDIR/TrackingRecHits ) mkdir $NEWREFDIR/TrackingRecHits
#if ( ! -e $NEWREFDIR/Tracks ) mkdir $NEWREFDIR/Tracks
#if ( ! -e $NEWREFDIR/TrackingParticles) mkdir $NEWREFDIR/TrackingParticles

cd ${DATADIR}



# Get the relevant packages
#
if (! -e Validation/Geometry) cvs co -r $CMSSW_VERSION Validation/Geometry
if (! -e Validation/TrackerHits) cvs co -r $CMSSW_VERSION Validation/TrackerHits
if (! -e Validation/TrackerDigis) cvs co -r $CMSSW_VERSION Validation/TrackerDigis
if (! -e Validation/TrackerRecHits) cvs co -r $CMSSW_VERSION Validation/TrackerRecHits
if (! -e Validation/TrackingMCTruth) cvs co -r $CMSSW_VERSION Validation/TrackingMCTruth
if (! -e Validation/RecoTrack) cvs co -r $CMSSW_VERSION Validation/RecoTrack
#Add also co of CalibTracker for test o Lite version of Geometry
if (! -e CalibTracker/SiStripCommon) cvs co -r $CMSSW_VERSION CalibTracker/SiStripCommon
#
# Geometry Validation
#
if ('${1}' == 'GEOMETRY' ) then
cd ${DATADIR}/Validation/Geometry/test
./TrackerGeometryValidation.sh ${RefRelease}
    if ($copyWWW == "true") then 
    chmod a+x copyWWWTrackerGeometry.sh    
    ./copyWWWTrackerGeometry.sh
    endif
endif
#
# Run validation chain
#
cd ${DATADIR}/Validation/TrackerConfiguration/test

#cp /afs/cern.ch/cms/data/CMSSW/Validation/TrackerHits/data/Muon.root .

if ($Mode == "Reconstruction") then
cmsRun Muon_FullChain_cfg.py

else if ($Mode == "Validation") then
cmsRun ValidationChainOnly_cfg.py

else if ($Mode == "Harvesting")then

cmsRun HarvestingGlobalValidation_Tracking.py
set histogramfile="DQM_V0001_R000000001__"$CMSSW_VERSION"__RelVal__Validation.root"
endif 

if ( ($Mode == "Comparison") || ($Mode == "Harvesting") ) then
cp $histogramfile TrackerHitHisto.root
cp $histogramfile stripdigihisto.root
cp $histogramfile pixeldigihisto.root
cp $histogramfile trackingtruthhisto.root
cp $histogramfile validationPlots.root
cp $histogramfile sistriprechitshisto.root
cp $histogramfile pixelrechitshisto.root
cp $histogramfile pixeltrackingrechitshist.root
cp $histogramfile striptrackingrechitshisto.root
cp $histogramfile $NEWREFDIR
endif

#if ( -e Muon.root ) /bin/rm Muon.root

cp ./TrackerHitHisto.root ${DATADIR}/Validation/TrackerHits/test/

cd ${DATADIR}/Validation/TrackerHits/test 

cp ${REFDIR}/SimHits/* ../

#cp TrackerHitHisto.root $NEWREFDIR/SimHit
if ( ! -e plots ) mkdir plots
root -b -p -q SiTrackerHitsCompareEnergy.C
if ( ! -e plots/muon ) mkdir plots/muon
gzip -f *.eps
mv eloss*.eps.gz plots/muon
mv eloss*.gif plots/muon

root -b -p -q SiTrackerHitsComparePosition.C
if ( ! -e plots/muon ) mkdir plots/muon
gzip -f *.eps
mv pos*.eps.gz plots/muon
mv pos*.gif plots/muon

root -b -p -q SiTrackerHitsComparePosition.C
if ( ! -e plots/muon ) mkdir plots/muon
gzip -f *.eps
mv pos*.eps.gz plots/muon
mv pos*.gif plots/muon

if ($copyWWW == "true") source copyWWWall.csh

cd ${DATADIR}/Validation/TrackerDigis/test 

cp ${DATADIR}/Validation/TrackerConfiguration/test/stripdigihisto.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/pixeldigihisto.root .

cp ${REFDIR}/Digis/* ../

#cp pixeldigihisto.root $NEWREFDIR/Digis
#cp stripdigihisto.root $NEWREFDIR/Digis

root -b -p -q  SiPixelDigiCompare.C
gzip -f *.eps
if ($copyWWW == "true") source copyWWWPixel.csh
root -b -p -q  SiStripDigiCompare.C
gzip -f *.eps
if ($copyWWW == "true") source copyWWWStrip.csh

cd ${DATADIR}/Validation/TrackerRecHits/test

cp ${DATADIR}/Validation/TrackerConfiguration/test/sistriprechitshisto.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/pixelrechitshisto.root .

cp ${REFDIR}/RecHits/* ../

#cp sistriprechitshisto.root $NEWREFDIR/RecHits/
#cp pixelrechitshisto.root $NEWREFDIR/RecHits/

root -b -p -q SiPixelRecHitsCompare.C
gzip -f *.eps
if ($copyWWW == "true") source copyWWWPixel.csh
root -b -p -q SiStripRecHitsCompare.C
gzip -f *.eps
if ($copyWWW == "true") source copyWWWStrip.csh

cd ${DATADIR}/Validation/TrackingMCTruth/test
cp ${DATADIR}/Validation/TrackerConfiguration/test/trackingtruthhisto.root .
cp ${REFDIR}/TrackingParticles/* ../
#cp trackingtruthhisto.root $NEWREFDIR/TrackingParticles/


root -b -p -q TrackingTruthCompare.C
gzip -f *.eps
if ($copyWWW == "true") source copyWWWTP.csh

cd ${DATADIR}/Validation/RecoTrack/test

cp ${DATADIR}/Validation/TrackerConfiguration/test/validationPlots.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/pixeltrackingrechitshist.root .
cp ${DATADIR}/Validation/TrackerConfiguration/test/striptrackingrechitshisto.root .

cp ${REFDIR}/Tracks/* ../
cp ${REFDIR}/TrackingRecHits/* ../

#cp validationPlots.root $NEWREFDIR/Tracks/
#cp pixeltrackingrechitshist.root $NEWREFDIR/TrackingRecHits/
#cp striptrackingrechitshisto.root $NEWREFDIR/TrackingRecHits/

root -b -p -q TracksCompareChain.C
gzip -f *.eps
if ($copyWWW == "true") source copyWWWTracks.csh

root -b -p -q SiPixelRecoCompare.C 
gzip -f *.eps
if ($copyWWW == "true") source copyWWWPixel.csh

root -b -p -q SiStripTrackingRecHitsCompare.C 
gzip -f *.eps
if ($copyWWW == "true") source copyWWWStrip.csh

#Check on the fly in order to check the correctness of LiteGeometry
#cd ${DATADIR}/CalibTracker/SiStripCommon/test
#cp ${REFDIR}/LiteGeometry/* oldgeometrylite.txt
#cmsRun writeFile.cfg
#cp myfile.txt $NEWREFDIR/LiteGeometry/

#diff myfile.txt oldgeometrylite.txt > ! litegeometryDIFF.txt
#mkdir /afs/cern.ch/cms/performance/tracker/activities/validation/${RefRelease}/LiteGeometry
#cp litegeometryDIFF.txt  /afs/cern.ch/cms/performance/tracker/activities/validation/${RefRelease}/LiteGeometry
