import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.PFElectronDQMAnalyzer_cfi import pfElectronDQMAnalyzer

pfElectronValidation1 = pfElectronDQMAnalyzer.clone()
#pfElectronValidation1.InputCollection = cms.InputTag('pfAllElectrons')
#pfElectronValidation1.MatchCollection = cms.InputTag('gensource')

#pfElectronValidation1.InputCollection = cms.InputTag('gsfElectrons','','PFlowDQM')
#pfElectronValidation1.InputCollection = cms.InputTag('mvaElectrons')
#pfElectronValidation1.InputCollection = cms.InputTag('particleFlow','','PFlowDQM')
#pfElectronValidation1.InputCollection = cms.InputTag('pfAllElectrons','','PFlowDQM')
pfElectronValidation1.InputCollection = cms.InputTag('pfAllElectrons') # for global Validation

#pfElectronValidation1.MatchCollection = cms.InputTag('genParticles')
#pfElectronValidation1.MatchCollection = cms.InputTag('g4SimHits')
#pfElectronValidation1.MatchCollection = cms.InputTag('gsfElectrons','','RECO')
#pfElectronValidation1.MatchCollection = cms.InputTag('particleFlow','','RECO')
#pfElectronValidation1.MatchCollection = cms.InputTag('genParticles','','RECO')
#pfElectronValidation1.MatchCollection = cms.InputTag('gensource','','PFlowDQM')
pfElectronValidation1.MatchCollection = cms.InputTag('gensource') # for global Validation

pfElectronValidation1.BenchmarkLabel  = cms.string('PFElectronValidation/CompWithGenElectron')
pfElectronValidationSequence = cms.Sequence( pfElectronValidation1 )


# NoTracking
pfElectronValidation2 = pfElectronDQMAnalyzer.clone()
#pfElectronValidation2.InputCollection = cms.InputTag('pfAllElectrons')
#pfElectronValidation2.MatchCollection = cms.InputTag('gensource')

#pfElectronValidation2.InputCollection = cms.InputTag('gsfElectrons','','RECO')
#pfElectronValidation2.InputCollection = cms.InputTag('mvaElectrons')
#pfElectronValidation2.InputCollection = cms.InputTag('particleFlow','','REPROD')
#pfElectronValidation2.InputCollection = cms.InputTag('particleFlow','','PFlowDQMnoTracking')
pfElectronValidation2.InputCollection = cms.InputTag('pfAllElectrons','','PFlowDQMnoTracking')

#pfElectronValidation2.MatchCollection = cms.InputTag('genParticles')
#pfElectronValidation2.MatchCollection = cms.InputTag('g4SimHits')
#pfElectronValidation2.MatchCollection = cms.InputTag('gsfElectrons','','PFlowDQM')
#pfElectronValidation2.MatchCollection = cms.InputTag('particleFlow','','RECO')
#pfElectronValidation2.MatchCollection = cms.InputTag('particleFlow','','PFlowDQM')
#pfElectronValidation2.MatchCollection = cms.InputTag('genParticles','','RECO')
pfElectronValidation2.MatchCollection = cms.InputTag('gensource','','PFlowDQMnoTracking')

pfElectronValidation2.BenchmarkLabel  = cms.string('PFElectronValidation/CompWithGenElectron')
pfElectronValidationSequence_NoTracking = cms.Sequence( pfElectronValidation2 )
