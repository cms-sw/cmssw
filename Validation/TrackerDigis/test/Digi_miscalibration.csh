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

#set usefakegain=_FakeGain
#set usefakela=_FakeLa
#set usefakenoise=_FakeNoise

set usefakegain 
set usefakela 
set usefakenoise

echo "set preferences"
cd ${DATADIR}

project CMSSW

# Get the relevant packages
#
#cvs co -r $CMSSW_VERSION Validation/TrackerDigis

cd ${DATADIR}/Validation/TrackerDigis/test 

#/// commentout if  reference are already there
cmsRun trackerdigivalid.cfg >& ! digi.log
mv pixeldigihisto.root $NEWREFDIR/Digis/Fake
mv stripdigihisto.root $NEWREFDIR/Digis/Fake
#//////

cp $NEWREFDIR/Digis/Fake/*.root ../data/

sed s/SCENARIO/IDEAL_V1/g trackerdigivalid_frontier.cfg >! tmp1.cfg

if ( $usefakegain == _FakeGain) then 
sed s/\#UNCOMMENTGAIN//g tmp1.cfg >! tmp2.cfg
else
cat tmp1.cfg >! tmp2.cfg
endif

if ( $usefakela == _FakeLa) then 
sed s/\#UNCOMMENTLA//g tmp2.cfg >! tmp3.cfg
else
cat tmp2.cfg >! tmp3.cfg
endif

if ( $usefakenoise == _FakeNoise) then 
sed s/\#UNCOMMENTNOISE//g tmp3.cfg >! Digi_ideal.cfg
else
cat tmp3.cfg >! Digi_ideal.cfg
endif
rm tmp*.cfg

cmsRun Digi_ideal.cfg >& ! digi.log

if($usefakegain == "" && $usefakela == "" && $usefakenoise== "") then
mv pixeldigihisto.root $NEWREFDIR/Digis/Ideal
mv stripdigihisto.root $NEWREFDIR/Digis/Ideal
cp $NEWREFDIR/Digis/Ideal/*.root .
endif

root -b -p -q  SiPixelDigiCompare.C
gzip *.eps
source copyWWWPixel.csh Fake_vs_Ideal$usefakenoise$usefakegain$usefakela
root -b -p -q  SiStripDigiCompare.C
gzip *.eps
source copyWWWStrip.csh Fake_vs_Ideal$usefakenoise$usefakegain$usefakela

mv pixeldigihisto.root ../data/
mv  stripdigihisto.root ../data/


sed s/SCENARIO/STARTUP/g trackerdigivalid_frontier.cfg >! tmp1.cfg

if ( $usefakegain == _FakeGain) then 
sed s/\#UNCOMMENTGAIN//g tmp1.cfg >! tmp2.cfg
else
cat tmp1.cfg >! tmp2.cfg
endif

if ( $usefakela == _Fake_La) then 
sed s/\#UNCOMMENTLA//g tmp2.cfg >! tmp3.cfg
else
cat tmp2.cfg >! tmp3.cfg
endif

if ( $usefakenoise == _FakeNoise) then 
sed s/\#UNCOMMENTNOISE//g tmp3.cfg >! Digi_startup.cfg
else
cat tmp3.cfg >! Digi_startup.cfg
endif
rm tmp*.cfg



cmsRun Digi_startup.cfg >& ! digi.log

if($usefakegain == "" && $usefakela == "" && $usefakenoise== "") then
cp pixeldigihisto.root $NEWREFDIR/Digis/Startup
cp stripdigihisto.root $NEWREFDIR/Digis/Startup
cp $NEWREFDIR/Digis/Startup/*.root . 
endif

root -b -p -q  SiPixelDigiCompare.C
gzip *.eps
source copyWWWPixel.csh Ideal_vs_Startup$usefakenoise$usefakegain$usefakela
root -b -p -q  SiStripDigiCompare.C
gzip *.eps
source copyWWWStrip.csh Ideal_vs_Startup$usefakenoise$usefakegain$usefakela


