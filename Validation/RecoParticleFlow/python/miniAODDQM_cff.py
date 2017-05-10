import FWCore.ParameterSet.Config as cms

# jet
from DQMOffline.PFTau.PFJetDQMAnalyzer_cfi import pfJetDQMAnalyzer

JetValidation1 = pfJetDQMAnalyzer.clone()
JetValidation1.BenchmarkLabel  = cms.string('slimmedJetValidation/CompWithPFJets')
JetValidation1.InputCollection = cms.InputTag('slimmedJets')
JetValidation1.MatchCollection = cms.InputTag('ak4PFJetsCHS') # ak5PFJetsCHS # ak5PFJets
JetValidation1.ptMin = cms.double(10.0)
JetValidation1.CreatePFractionHistos = cms.bool(True)
#JetValidation1.InputCollection = cms.InputTag('ak5PFJets')
#JetValidation1.MatchCollection = cms.InputTag('slimmedJets')

JetValidation2 = pfJetDQMAnalyzer.clone()
JetValidation2.BenchmarkLabel  = cms.string('slimmedJetValidation/CompWithPFJetsEC')
#JetValidation2.InputCollection = JetValidation1.MatchCollection
#JetValidation2.MatchCollection = JetValidation1.InputCollection
JetValidation2.InputCollection = cms.InputTag('slimmedJets')
JetValidation2.MatchCollection = cms.InputTag('ak4PFJetsNewL1Fast23') # ak4PFJetsCHSEC # ak4PFJetsCHS
JetValidation2.ptMin = JetValidation1.ptMin
JetValidation2.CreatePFractionHistos = cms.bool(True)


# jetRes plots
from DQMOffline.PFTau.PFJetResDQMAnalyzer_cfi import pfJetResDQMAnalyzer

JetResValidation1 = pfJetResDQMAnalyzer.clone()
JetResValidation1.InputCollection = JetValidation1.InputCollection
JetResValidation1.MatchCollection = JetValidation1.MatchCollection
JetResValidation1.ptMin = JetValidation1.ptMin

JetResValidation2 = pfJetResDQMAnalyzer.clone()
JetResValidation2.InputCollection = JetValidation2.InputCollection
JetResValidation2.MatchCollection = JetValidation2.MatchCollection
JetResValidation2.ptMin = JetValidation2.ptMin


# MET
from DQMOffline.PFTau.PFMETDQMAnalyzer_cfi import pfMETDQMAnalyzer

METValidation1 = pfMETDQMAnalyzer.clone()
METValidation1.BenchmarkLabel  = cms.string('slimmedMETValidation/CompWithPFMET')
METValidation1.InputCollection = cms.InputTag('slimmedMETs')
METValidation1.MatchCollection = cms.InputTag('pfMet')

METValidation2 = pfMETDQMAnalyzer.clone()
METValidation2.BenchmarkLabel  = cms.string('slimmedMETValidation/CompWithPFMETT1')
METValidation2.InputCollection = cms.InputTag('slimmedMETs')
METValidation2.MatchCollection = cms.InputTag('pfMetT1')


# muons
from DQMOffline.PFTau.PFMuonDQMAnalyzer_cfi import pfMuonDQMAnalyzer

slimmedMuonValidation1 = pfMuonDQMAnalyzer.clone()
slimmedMuonValidation1.BenchmarkLabel  = cms.string('SlimmedMuonValidation/CompWithRecoMuons')
slimmedMuonValidation1.InputCollection = cms.InputTag('slimmedMuons')
slimmedMuonValidation1.MatchCollection = cms.InputTag('muons')
# official
#muonPFsequenceMC.inputTagMuonReco = cms.InputTag('slimmedMuons')
#muonPFsequenceMC.inputTagGenParticles = cms.InputTag('muons')
#muonPFsequenceMC.runOnMC = cms.bool(False)
# RefCore: A request to resolve a reference to a product of type 'std::vector<reco::Track>' with ProductID '3:1469' can not be satisfied because the product cannot be found.
# with the following:
#muonPFsequenceMC.inputTagMuonReco = cms.InputTag('muons')
#muonPFsequenceMC.inputTagGenParticles = cms.InputTag('slimmedMuons')


# electrons
from DQMOffline.PFTau.PFElectronDQMAnalyzer_cfi import pfElectronDQMAnalyzer

ElectronValidation1 = pfElectronDQMAnalyzer.clone()
ElectronValidation1.BenchmarkLabel  = cms.string('slimmedElectronValidation/CompWithGedGsfElectrons')
ElectronValidation1.InputCollection = cms.InputTag('slimmedElectrons')
ElectronValidation1.MatchCollection = cms.InputTag('gedGsfElectrons') 
# use electrons plots for muons
#ElectronValidation2 = pfElectronDQMAnalyzer.clone()
#ElectronValidation2.BenchmarkLabel  = slimmedMuonValidation1.BenchmarkLabel
#ElectronValidation2.InputCollection = slimmedMuonValidation1.InputCollection
#ElectronValidation2.MatchCollection = slimmedMuonValidation1.MatchCollection



miniAODDQMSequence = cms.Sequence(
                                  JetValidation1 * JetValidation2 *
                                  JetResValidation1 * JetResValidation2 *
                                  METValidation1 * METValidation2 *
                                  slimmedMuonValidation1 *
                                  ElectronValidation1
                                  )
