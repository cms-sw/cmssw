import FWCore.ParameterSet.Config as cms


from DQMOffline.PFTau.PFJetDQMAnalyzer_cfi import pfJetDQMAnalyzer

pfJetValidation1 = pfJetDQMAnalyzer.clone()
pfJetValidation1.InputCollection = cms.InputTag('ak4PFJets')
pfJetValidation1.MatchCollection = cms.InputTag('ak4GenJets')
pfJetValidation1.BenchmarkLabel  = cms.string('PFJetValidation/CompWithGenJet')

pfJetValidation2 = pfJetDQMAnalyzer.clone()
pfJetValidation2.InputCollection = cms.InputTag('ak4PFJets')
pfJetValidation2.MatchCollection = cms.InputTag('ak4CaloJets')
pfJetValidation2.BenchmarkLabel  = cms.string('PFJetValidation/CompWithCaloJet')

pfJetValidationSequence = cms.Sequence( pfJetValidation1 * pfJetValidation2 )

