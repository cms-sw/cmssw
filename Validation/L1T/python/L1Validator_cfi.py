import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
L1Validator = DQMEDAnalyzer('L1Validator',
  dirName=cms.string("L1T/L1TriggerVsGen"),
#  fileName=cms.string("L1Validation.root") #output file name
  GenSource=cms.InputTag("genParticles"),
  srcToken = cms.InputTag("generator"),
  L1MuonBXSource=cms.InputTag("gmtStage2Digis", "Muon"),
  L1EGammaBXSource=cms.InputTag("caloStage2Digis", "EGamma"),
  L1TauBXSource=cms.InputTag("caloStage2Digis", "Tau"),
  L1JetBXSource=cms.InputTag("caloStage2Digis", "Jet"),
  L1ExtraMuonSource=cms.InputTag("l1extraParticles"),
  L1GenJetSource=cms.InputTag("ak4GenJets","")
)
