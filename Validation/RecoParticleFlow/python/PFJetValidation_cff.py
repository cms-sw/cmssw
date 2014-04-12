import FWCore.ParameterSet.Config as cms


from DQMOffline.PFTau.PFJetDQMAnalyzer_cfi import pfJetDQMAnalyzer

pfJetValidation1 = pfJetDQMAnalyzer.clone()
pfJetValidation1.InputCollection = cms.InputTag('ak5PFJets')
pfJetValidation1.MatchCollection = cms.InputTag('ak5GenJets')
pfJetValidation1.BenchmarkLabel  = cms.string('PFJetValidation/CompWithGenJet')

pfJetValidation2 = pfJetDQMAnalyzer.clone()
pfJetValidation2.InputCollection = cms.InputTag('ak5PFJets')
pfJetValidation2.MatchCollection = cms.InputTag('ak5CaloJets')
pfJetValidation2.BenchmarkLabel  = cms.string('PFJetValidation/CompWithCaloJet')
pfJetValidation2.SkimParameter.switchOn  = cms.bool(False)

pfJetValidationSequence = cms.Sequence( pfJetValidation1 * pfJetValidation2 )

