#! /bin/csh

eval `scramv1 runtime -csh`

setenv DATADIR $CMSSW_BASE/src
setenv REFDIRS /afs/cern.ch/cms/performance/tracker/activities/validation/ReferenceFiles

setenv NEWREFDIR $REFDIRS/$CMSSW_VERSION
if ( ! -e $NEWREFDIR ) mkdir $NEWREFDIR
if ( ! -e $NEWREFDIR/Digis ) mkdir $NEWREFDIR/Digis
if ( ! -e $NEWREFDIR/Digis/Startup ) mkdir $NEWREFDIR/Digis/Startup
if ( ! -e $NEWREFDIR/Digis/Ideal ) mkdir $NEWREFDIR/Digis/Ideal
if ( ! -e $NEWREFDIR/Digis/Fake ) mkdir $NEWREFDIR/Digis/Fake

cd ${DATADIR}

project CMSSW

# Get the relevant packages
#
cvs co -r $CMSSW_VERSION Validation/TrackerDigis

cd ${DATADIR}/Validation/TrackerDigis/test 

#cmsRun trackerdigivalid.cfg >& ! digi.log
#mv pixeldigihisto.root $NEWREFDIR/Digis/Fake
#mv stripdigihisto.root $NEWREFDIR/Digis/Fake
cp $NEWREFDIR/Digis/Fake/*.root ../data/

sed s/SCENARIO/IDEAL_V1/g trackerdigivalid_frontier.cfg >! Digi_ideal.cfg

#cmsRun Digi_ideal.cfg >& ! digi.log

#mv pixeldigihisto.root $NEWREFDIR/Digis/Ideal
#mv stripdigihisto.root $NEWREFDIR/Digis/Ideal
cp $NEWREFDIR/Digis/Ideal/*.root .

root -b -p -q  SiPixelDigiCompare.C
gzip *.eps
source copyWWWPixel.csh Fake_vs_Ideal
root -b -p -q  SiStripDigiCompare.C
gzip *.eps
source copyWWWStrip.csh Fake_vs_Ideal

mv pixeldigihisto.root ../data/
mv  stripdigihisto.root ../data/


sed s/SCENARIO/STARTUP/g trackerdigivalid_frontier.cfg >! Digi_startup.cfg

#cmsRun Digi_startup.cfg >& ! digi.log
#cp pixeldigihisto.root $NEWREFDIR/Digis/Startup
#cp stripdigihisto.root $NEWREFDIR/Digis/Startup

cp $NEWREFDIR/Digis/Startup/*.root . 

root -b -p -q  SiPixelDigiCompare.C
gzip *.eps
source copyWWWPixel.csh Ideal_vs_Startup
root -b -p -q  SiStripDigiCompare.C
gzip *.eps
source copyWWWStrip.csh Ideal_vs_Startup

