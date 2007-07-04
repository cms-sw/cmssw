#! /bin/csh

setenv DATADIR  /afs/cern.ch/cms/data/CMSSW/Validation/TrackerHits/data

eval `scramv1 ru -csh`

 echo "===========> Validating Tracker Simhits with 15 GeV Muon eta ......."
 cp  ${DATADIR}/Muon.root   .

 cmsRun  TrackerHitValid.cfg >& output.log
 /bin/rm output.log

 if ( ! -e plots ) mkdir plots

 root -b -p -q SiStripHitsCompareEnergy.C
 if ( ! -e plots/muon ) mkdir plots/muon
 /bin/mv eloss*.eps plots/muon
 /bin/mv eloss*.gif plots/muon

 root -b -p -q SiStripHitsComparePosition.C
 if ( ! -e plots/muon ) mkdir plots/muon
 /bin/mv pos*.eps plots/muon
 /bin/mv pos*.gif plots/muon

 if ( -e Muon.root ) /bin/rm Muon.root
  
echo "...Done..."
