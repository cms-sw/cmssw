import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.PFElectronDQMAnalyzer_cfi import pfElectronDQMAnalyzer

pfElectronValidation1 = pfElectronDQMAnalyzer.clone()
pfElectronValidation1.InputCollection = cms.InputTag('pfAllElectrons') # for global Validation
pfElectronValidation1.MatchCollection = cms.InputTag('gensource') # for global Validation
pfElectronValidation1.BenchmarkLabel  = cms.string('PFElectronValidation/CompWithGenElectron')
pfElectronValidationSequence = cms.Sequence( pfElectronValidation1 )


# NoTracking
pfElectronValidation2 = pfElectronDQMAnalyzer.clone()
pfElectronValidation2.InputCollection = cms.InputTag('pfAllElectrons','','PFlowDQMnoTracking')
pfElectronValidation2.MatchCollection = cms.InputTag('gensource','','PFlowDQMnoTracking')
pfElectronValidation2.BenchmarkLabel  = cms.string('PFElectronValidation/CompWithGenElectron')
pfElectronValidationSequence_NoTracking = cms.Sequence( pfElectronValidation2 )
