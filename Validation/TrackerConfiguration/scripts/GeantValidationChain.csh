#! /bin/csh

eval `scramv1 runtime -csh`

setenv DATADIR $CMSSW_BASE/src

setenv REFDIRS /afs/cern.ch/cms/performance/tracker/activities/validation/ReferenceFiles
setenv REFDIR $REFDIRS/${1}
setenv NEWREFDIR $REFDIRS/$CMSSW_VERSION
if ( ! -e $NEWREFDIR ) mkdir $NEWREFDIR
if ( ! -e $NEWREFDIR/SimHits ) mkdir $NEWREFDIR/SimHits

cd ${DATADIR}

project CMSSW
#
# Get the relevant packages
#
#cvs co -r $CMSSW_VERSION Validation/TrackerHits
cvs co Validation/TrackerHits
#
# Run simhit validation on 1GeV electrons (QGSP_EMV)
#
cd ${DATADIR}/Validation/TrackerConfiguration/test

cp /afs/cern.ch/cms/performance/tracker/activities/validation/G4Validation/Electron.root ./Muon.root

cmsRun SimHits_Validation_cfg.py
cp ./TrackerHitHisto.root  TrackerHitHisto_ele_QGSP_EMV.root
cp ./TrackerHitHisto.root ${DATADIR}/Validation/TrackerHits/test/

mv TrackerHitHisto_ele_QGSP_EMV.root $NEWREFDIR/SimHits/

cd ${DATADIR}/Validation/TrackerHits/test 

if($2 == 1) then
cp ${REFDIR}/SimHits/TrackerHitHisto_ele_QGSP_EMV.root ../data/TrackerHitHisto.root

if ( ! -e plots ) mkdir plots
root -b -p -q SiStripHitsCompareEnergy.C
if ( ! -e plots/muon ) mkdir plots/muon
mv eloss*.eps plots/muon
mv eloss*.gif plots/muon

root -b -p -q SiStripHitsComparePosition.C
if ( ! -e plots/muon ) mkdir plots/muon
mv pos*.eps plots/muon
mv pos*.gif plots/muon

source copyWWWall_geant.csh ele_QGSP_EMV
end

#
# Run simhit validation on 1GeV pions (QGSP_EMV)
#
cd ${DATADIR}/Validation/TrackerConfiguration/test

cp /afs/cern.ch/cms/performance/tracker/activities/validation/G4Validation/Pion.root ./Muon.root

cmsRun SimHits_Validation.cfg

cp ./TrackerHitHisto.root TrackerHitHisto_pi_QGSP_EMV.root
cp ./TrackerHitHisto.root ${DATADIR}/Validation/TrackerHits/test/

mv TrackerHitHisto_pi_QGSP_EMV.root $NEWREFDIR/SimHits/
cd ${DATADIR}/Validation/TrackerHits/test 

if($1 == 1) then
cp ${REFDIR}/SimHits/TrackerHitHisto_pi_QGSP_EMV.root ../data/TrackerHitHisto.root

if ( ! -e plots ) mkdir plots
root -b -p -q SiStripHitsCompareEnergy.C
if ( ! -e plots/muon ) mkdir plots/muon
mv eloss*.eps plots/muon
mv eloss*.gif plots/muon

root -b -p -q SiStripHitsComparePosition.C
if ( ! -e plots/muon ) mkdir plots/muon
mv pos*.eps plots/muon
mv pos*.gif plots/muon

source copyWWWall_geant.csh pi_QGSP_EMV
end
