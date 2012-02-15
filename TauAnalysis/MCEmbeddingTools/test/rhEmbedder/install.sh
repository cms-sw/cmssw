#! /bin/bash

echo 
echo
echo Warning. All changes made locally in your dev area will be lost in 10 seconds
echo 
echo
sleep 10s

wd=`pwd`
targetDir=$CMSSW_BASE/src/
cp DetAssoc.patch $targetDir
cd $targetDir
addpkg TrackingTools/TrackAssociator
patch -p0 -i DetAssoc.patch

# taken from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideTauAnalysis?redirectedfrom=CMS.SWGuideTauAnalysis#CMSSW_4_2_X
echo "checking-out RecoTauTag packages"
cvs co -r 1.49 RecoTauTag/tau_tags.txt
addpkg -f RecoTauTag/tau_tags.txt

echo "checking-out PAT packages"
cvs co -r CMSSW_4_2_4_patch1 PhysicsTools/MVAComputer    
cvs up -r 1.3 PhysicsTools/MVAComputer/interface/MVAModuleHelper.h
cvs co -r V08-06-36 PhysicsTools/PatAlgos
cvs up -r 1.47 PhysicsTools/PatAlgos/python/tools/tauTools.py
cvs up -r 1.5 PhysicsTools/PatAlgos/plugins/PATSingleVertexSelector.cc
cvs up -r 1.5 PhysicsTools/PatAlgos/plugins/PATSingleVertexSelector.h
cvs co -r V08-03-12 PhysicsTools/Utilities
cvs co -r CMSSW_4_2_4_patch1 DataFormats/PatCandidates
cvs up -r 1.32 DataFormats/PatCandidates/interface/Tau.h
cvs up -r 1.3 DataFormats/PatCandidates/interface/TauCaloSpecific.h
cvs up -r 1.7 DataFormats/PatCandidates/interface/TauPFSpecific.h
cvs up -r 1.21 DataFormats/PatCandidates/src/Tau.cc
cvs up -r 1.3 DataFormats/PatCandidates/src/TauCaloSpecific.cc
cvs up -r 1.6 DataFormats/PatCandidates/src/TauPFSpecific.cc
cvs co -r 1.1 DataFormats/PatCandidates/interface/TauJetCorrFactors.h
cvs co -r 1.1 DataFormats/PatCandidates/src/TauJetCorrFactors.cc
cvs up -r 1.60 DataFormats/PatCandidates/src/classes.h
cvs up -r 1.70 DataFormats/PatCandidates/src/classes_def.xml

echo "checking-out TauAnalysis packages and other packages required by TauAnalysis" 
cvs co -r V00-01-01 CommonTools/CandUtils
cvs co -r cbern_isolation_29Sept11_d CommonTools/ParticleFlow
cvs up -r 1.1.2.2 CommonTools/ParticleFlow/python/pfPileUpCandidates_cff.py
cvs up -r 1.2 CommonTools/ParticleFlow/python/pfPileUp_cfi.py
cvs up -r 1.2 CommonTools/ParticleFlow/python/TopProjectors/pfNoPileUp_cfi.py
cvs up -r 1.2 CommonTools/ParticleFlow/python/Isolation/pfMuonIsolationFromDeposits_cff.py
cvs up -r 1.2 CommonTools/ParticleFlow/python/ParticleSelectors/pfCandsForIsolation_cff.py
cvs co -r V00-00-01 HiggsAnalysis/HiggsToTauTau
cvs co -r V00-06-08 MuonAnalysis/MomentumScaleCalibration     
cvs co -r CMSSW_4_2_4_patch1 PhysicsTools/CandUtils
cvs up -r 1.3 PhysicsTools/CandUtils/src/EventShapeVariables.cc
cvs co -r V04-05-07 JetMETCorrections/Type1MET 
cvs co -r b4_2_X_cvMEtCorr_02Feb2012 PhysicsTools/PatUtils 
cvs co -r V02-03-00 JetMETCorrections/Algorithms                     
rm JetMETCorrections/Algorithms/interface/L1JPTOffsetCorrector.h
rm JetMETCorrections/Algorithms/src/L1JPTOffsetCorrector.cc
cvs co -r V03-01-00 JetMETCorrections/Objects        
cvs co -r CMSSW_4_2_4_patch1 DataFormats/METReco
cvs up -r 1.28 DataFormats/METReco/src/classes.h
cvs up -r 1.25 DataFormats/METReco/src/classes_def.xml
cvs co -r cbern_isolation_29Sept11_b RecoMuon/MuonIsolation
cp /afs/cern.ch/user/v/veelken/public/MuPFIsoHelper.cc RecoMuon/MuonIsolation/src
cvs co -r CMSSW_4_2_4_patch1 DataFormats/MuonReco
cvs up -r 1.1 DataFormats/MuonReco/interface/MuonPFIsolation.h
cvs co -r b4_2_x_2012Feb02 TauAnalysis
cvs up -r 1.5 TauAnalysis/CandidateTools/interface/VBFCompositePtrCandidateT1T2MEtEventT3Producer.h
#cvs co -r b4_2_x_2012Feb02 AnalysisDataFormats/TauAnalysis
rm -rf TauAnalysis/MCEmbeddingTools/
cvs co TauAnalysis/MCEmbeddingTools
rm -rf TauAnalysis/MCEmbeddingTools/test/rhEmbedder/install.sh


cvs co TauAnalysis/Skimming/python/goldenZmmSelectionVBTFrelPFIsolation_cfi.py

scramv1 b -j 12
