import FWCore.ParameterSet.Config as cms


from DQMOffline.PFTau.PFMETDQMAnalyzer_cfi import pfMETDQMAnalyzer

pfMETValidation1 = pfMETDQMAnalyzer.clone()
pfMETValidation1.InputCollection = cms.InputTag('pfMet')
pfMETValidation1.MatchCollection = cms.InputTag('genMetTrue')
pfMETValidation1.BenchmarkLabel  = cms.string('PFMETValidation/CompWithGenMET')

pfMETValidation2 = pfMETDQMAnalyzer.clone()
pfMETValidation2.InputCollection = cms.InputTag('pfMet')
pfMETValidation2.MatchCollection = cms.InputTag('caloMet')
pfMETValidation2.BenchmarkLabel  = cms.string('PFMETValidation/CompWithCaloMET')
pfMETValidation2.SkimParameter.switchOn  = cms.bool(False)

pfMETValidationSequence = cms.Sequence( pfMETValidation1 * pfMETValidation2 )
