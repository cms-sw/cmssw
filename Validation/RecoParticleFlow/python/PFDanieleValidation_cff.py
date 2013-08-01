import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.PFDanieleDQMAnalyzer_cfi import pfDanieleDQMAnalyzer

pfDanieleValidation1 = pfDanieleDQMAnalyzer.clone()
#pfDanieleValidation1.InputCollection = cms.InputTag('pfAllElectrons')
#pfDanieleValidation1.MatchCollection = cms.InputTag('gensource')

#pfDanieleValidation1.InputCollection = cms.InputTag('gsfElectrons','','PFlowDQM')
#pfDanieleValidation1.InputCollection = cms.InputTag('mvaElectrons')
#pfDanieleValidation1.InputCollection = cms.InputTag('particleFlow','','PFlowDQM')

#pfDanieleValidation1.InputCollection = cms.InputTag('ak5PFJets','','PFlowDQM')
pfDanieleValidation1.InputCollection = cms.InputTag('ak5PFJets') # for global Validation

#pfDanieleValidation1.MatchCollection = cms.InputTag('genParticles')
#pfDanieleValidation1.MatchCollection = cms.InputTag('g4SimHits')
#pfDanieleValidation1.MatchCollection = cms.InputTag('gsfElectrons','','RECO')
#pfDanieleValidation1.MatchCollection = cms.InputTag('particleFlow','','RECO')

#pfDanieleValidation1.MatchCollection = cms.InputTag('ak5GenJets','','PFlowDQM')
pfDanieleValidation1.MatchCollection = cms.InputTag('ak5GenJets') # for global Validatio


pfDanieleValidation1.BenchmarkLabel  = cms.string('ElectronValidation/JetPtRes')
pfDanieleValidationSequence = cms.Sequence( pfDanieleValidation1 )

# NoTracking
pfDanieleValidation2 = pfDanieleDQMAnalyzer.clone()
#pfDanieleValidation2.InputCollection = cms.InputTag('pfAllElectrons')
#pfDanieleValidation2.MatchCollection = cms.InputTag('gensource')

#pfDanieleValidation2.InputCollection = cms.InputTag('gsfElectrons','','RECO')
#pfDanieleValidation2.InputCollection = cms.InputTag('mvaElectrons')
#pfDanieleValidation2.InputCollection = cms.InputTag('particleFlow','','REPROD')
#pfDanieleValidation2.InputCollection = cms.InputTag('particleFlow','','PFlowDQMnoTracking')
pfDanieleValidation2.InputCollection = cms.InputTag('ak5PFJets','','PFlowDQMnoTracking')

#pfDanieleValidation2.MatchCollection = cms.InputTag('genParticles')
#pfDanieleValidation2.MatchCollection = cms.InputTag('g4SimHits')
#pfDanieleValidation2.MatchCollection = cms.InputTag('gsfElectrons','','PFlowDQM')
#pfDanieleValidation2.MatchCollection = cms.InputTag('particleFlow','','RECO')
#pfDanieleValidation2.MatchCollection = cms.InputTag('particleFlow','','PFlowDQM')
pfDanieleValidation2.MatchCollection = cms.InputTag('ak5GenJets','','PFlowDQMnoTracking')

pfDanieleValidation2.BenchmarkLabel  = cms.string('ElectronValidation/JetPtRes')
pfDanieleValidationSequence_NoTracking = cms.Sequence( pfDanieleValidation2 )

