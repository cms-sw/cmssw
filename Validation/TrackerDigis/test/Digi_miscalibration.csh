#! /bin/csh

eval `scramv1 runtime -csh`

setenv DATADIR $CMSSW_BASE/src
setenv REFDIRS /afs/cern.ch/cms/performance/tracker/activities/validation/ReferenceFiles
setenv IDEALTAG IDEAL_V1
setenv STARTUPTAG STARTUP
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
cmsRun trackerdigivalid_cfg.py >& ! digi.log
mv pixeldigihisto.root $NEWREFDIR/Digis/Fake
mv stripdigihisto.root $NEWREFDIR/Digis/Fake
#//////

cp $NEWREFDIR/Digis/Fake/*.root ../data/

sed s/SCENARIO/$IDEALTAG/g trackerdigivalid_frontier_cfg.py >! tmp1_cfg.py

if ( $usefakegain == _FakeGain) then 
sed s/\#UNCOMMENTGAIN//g tmp1_cfg.py >! tmp2_cfg.py
else
cat tmp1_cfg.py >! tmp2_cfg.py
endif

if ( $usefakela == _FakeLa) then 
sed s/\#UNCOMMENTLA//g tmp2_cfg.py >! tmp3_cfg.py
else
cat tmp2_cfg.py >! tmp3_cfg.py
endif

if ( $usefakenoise == _FakeNoise) then 
sed s/\#UNCOMMENTNOISE//g tmp3_cfg.py >! Digi_ideal_cfg.py
else
cat tmp3_cfg.py >! Digi_ideal_cfg.py
endif
rm tmp*_cfg.py

cmsRun Digi_ideal_cfg.py >& ! digi.log

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


sed s/SCENARIO/$STARTUPTAG/g trackerdigivalid_frontier_cfg.py >! tmp1_cfg.py

if ( $usefakegain == _FakeGain) then 
sed s/\#UNCOMMENTGAIN//g tmp1_cfg.py >! tmp2_cfg.py
else
cat tmp1_cfg.py >! tmp2_cfg.py
endif

if ( $usefakela == _Fake_La) then 
sed s/\#UNCOMMENTLA//g tmp2_cfg.py >! tmp3_cfg.py
else
cat tmp2_cfg.py >! tmp3_cfg.py
endif

if ( $usefakenoise == _FakeNoise) then 
sed s/\#UNCOMMENTNOISE//g tmp3_cfg.py >! Digi_startup_cfg.py
else
cat tmp3_cfg.py >! Digi_startup_cfg.py
endif
rm tmp*_cfg.py



cmsRun Digi_startup_cfg.py >& ! digi.log

if($usefakegain == "" && $usefakela == "" && $usefakenoise== "") then
cp pixeldigihisto.root $NEWREFDIR/Digis/Startup
cp stripdigihisto.root $NEWREFDIR/Digis/Startup
cp $NEWREFDIR/Digis/Startup/*.root . 
endif

root -b -p -q  SiPixelDigiCompare.C
gzip *.eps
source copyWWWPixel.csh dummy Ideal_vs_Startup$usefakenoise$usefakegain$usefakela
root -b -p -q  SiStripDigiCompare.C
gzip *.eps
source copyWWWStrip.csh dummy Ideal_vs_Startup$usefakenoise$usefakegain$usefakela


