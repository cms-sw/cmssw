import FWCore.ParameterSet.Config as cms

from DQMOffline.PFTau.PFJetResDQMAnalyzer_cfi import pfJetResDQMAnalyzer

pfJetResValidation1 = pfJetResDQMAnalyzer.clone()
#pfJetResValidation1.InputCollection = cms.InputTag('pfAllElectrons')
#pfJetResValidation1.MatchCollection = cms.InputTag('gensource')

#pfJetResValidation1.InputCollection = cms.InputTag('gsfElectrons','','PFlowDQM')
#pfJetResValidation1.InputCollection = cms.InputTag('mvaElectrons')
#pfJetResValidation1.InputCollection = cms.InputTag('particleFlow','','PFlowDQM')

#pfJetResValidation1.InputCollection = cms.InputTag('ak5PFJets','','PFlowDQM')
pfJetResValidation1.InputCollection = cms.InputTag('ak5PFJets') # for global Validation

#pfJetResValidation1.MatchCollection = cms.InputTag('genParticles')
#pfJetResValidation1.MatchCollection = cms.InputTag('g4SimHits')
#pfJetResValidation1.MatchCollection = cms.InputTag('gsfElectrons','','RECO')
#pfJetResValidation1.MatchCollection = cms.InputTag('particleFlow','','RECO')

#pfJetResValidation1.MatchCollection = cms.InputTag('ak5GenJets','','PFlowDQM')
pfJetResValidation1.MatchCollection = cms.InputTag('ak5GenJets') # for global Validatio


pfJetResValidation1.BenchmarkLabel  = cms.string('ElectronValidation/JetPtRes')
pfJetResValidationSequence = cms.Sequence( pfJetResValidation1 )

# NoTracking
pfJetResValidation2 = pfJetResDQMAnalyzer.clone()
#pfJetResValidation2.InputCollection = cms.InputTag('pfAllElectrons')
#pfJetResValidation2.MatchCollection = cms.InputTag('gensource')

#pfJetResValidation2.InputCollection = cms.InputTag('gsfElectrons','','RECO')
#pfJetResValidation2.InputCollection = cms.InputTag('mvaElectrons')
#pfJetResValidation2.InputCollection = cms.InputTag('particleFlow','','REPROD')
#pfJetResValidation2.InputCollection = cms.InputTag('particleFlow','','PFlowDQMnoTracking')
pfJetResValidation2.InputCollection = cms.InputTag('ak5PFJets','','PFlowDQMnoTracking')

#pfJetResValidation2.MatchCollection = cms.InputTag('genParticles')
#pfJetResValidation2.MatchCollection = cms.InputTag('g4SimHits')
#pfJetResValidation2.MatchCollection = cms.InputTag('gsfElectrons','','PFlowDQM')
#pfJetResValidation2.MatchCollection = cms.InputTag('particleFlow','','RECO')
#pfJetResValidation2.MatchCollection = cms.InputTag('particleFlow','','PFlowDQM')
pfJetResValidation2.MatchCollection = cms.InputTag('ak5GenJets','','PFlowDQMnoTracking')

pfJetResValidation2.BenchmarkLabel  = cms.string('ElectronValidation/JetPtRes')
pfJetResValidationSequence_NoTracking = cms.Sequence( pfJetResValidation2 )

