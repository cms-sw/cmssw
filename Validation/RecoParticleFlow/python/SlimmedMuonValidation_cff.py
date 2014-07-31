import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.PFMuonDQMAnalyzer_cfi import pfMuonDQMAnalyzer

slimmedMuonValidation1 = pfMuonDQMAnalyzer.clone()
slimmedMuonValidation1.BenchmarkLabel  = cms.string('SlimmedMuonValidation/CompWithRecoMuon')
slimmedMuonValidation1.InputCollection = cms.InputTag('slimmedMuons')
slimmedMuonValidation1.MatchCollection = cms.InputTag('muons')

slimmedMuonValidationSequence = cms.Sequence( slimmedMuonValidation1 )
