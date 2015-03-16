import FWCore.ParameterSet.Config as cms

# Basic HepMC/GenParticle/GenJet validation modules
from Validation.EventGenerator.BasicHepMCValidation_cfi import *
from Validation.EventGenerator.BasicHepMCHeavyIonValidation_cfi import *
from Validation.EventGenerator.BasicGenParticleValidation_cfi import *

# Analyzer for MB/UE studies
from Validation.EventGenerator.MBUEandQCDValidation_cff import *

# Duplication Checker, for LHE workflows
from Validation.EventGenerator.DuplicationChecker_cfi import *

# simple analyzer for DrellYan->lepton processes
from Validation.EventGenerator.DrellYanValidation_cff import *

# simple analyzer for W->lepton processes
from Validation.EventGenerator.WValidation_cff import *

# simple analyzer for tau decays validation
from Validation.EventGenerator.TauValidation_cfi import *

#TTbar Analyzer
from Validation.EventGenerator.TTbar_Validation_cfi import *

#Higgs
from Validation.EventGenerator.HiggsValidation_cfi  import *

#B-physics
from Validation.EventGenerator.BPhysicsValidation_cfi  import *

# define sequences...
basicGenTest_seq = cms.Sequence(basicHepMCValidation+basicGenParticleValidation)
duplicationChecker_seq = cms.Sequence(duplicationChecker)
mbueAndqcdValidation_seq = cms.Sequence(mbueAndqcd_seq)
drellYanValidation_seq = cms.Sequence(drellYanEleValidation+drellYanMuoValidation)
wValidation_seq = cms.Sequence(wEleValidation+wMuoValidation)
tauValidation_seq = cms.Sequence(tauValidation)
genLeptons_seq = cms.Sequence(genParticlesShortList*genParticlesMuons*genParticlesElectrons*genParticlesNeutrinos)
analyzeGenLeptons_seq = cms.Sequence(analyzeGenMuons*analyzeGenElecs*analyzeGenNtrns)
TTbarfull_seq = cms.Sequence(TTbarAnalyzeSpinCorr*analyzeTopKinematics*genLeptons_seq*analyzeGenLeptons_seq*analyzeGenJets)
bphysics = cms.Sequence(JPsiMuMuValidation*LambdabPiPiMuMuValidation*LambdaSpectrum*PsiSpectrum)
# master sequences for different processes/topologies validation

genvalid = cms.Sequence(basicGenTest_seq)
genvalid_qcd = cms.Sequence(basicGenTest_seq+mbueAndqcdValidation_seq)
genvalid_dy = cms.Sequence(basicGenTest_seq+mbueAndqcdValidation_seq+drellYanValidation_seq+tauValidation_seq)
genvalid_w = cms.Sequence(basicGenTest_seq+mbueAndqcdValidation_seq+wValidation_seq+tauValidation_seq)
genvalid_all = cms.Sequence(basicGenTest_seq+mbueAndqcdValidation_seq+drellYanValidation_seq+wValidation_seq+tauValidation_seq+TTbarfull_seq+higgsValidation+bphysics)
genvalid_all_and_dup_check = cms.Sequence(duplicationChecker_seq+genvalid_all)
