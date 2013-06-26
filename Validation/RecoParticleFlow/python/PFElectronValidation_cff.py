import FWCore.ParameterSet.Config as cms


from DQMOffline.PFTau.PFElectronDQMAnalyzer_cfi import pfElectronDQMAnalyzer

pfElectronValidation1 = pfElectronDQMAnalyzer.clone()
pfElectronValidation1.InputCollection = cms.InputTag('pfAllElectrons')
pfElectronValidation1.MatchCollection = cms.InputTag('gensource')
pfElectronValidation1.BenchmarkLabel  = cms.string('PFElectronValidation/CompWithGenElectron')

pfElectronValidationSequence = cms.Sequence( pfElectronValidation1 )

