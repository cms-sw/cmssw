import FWCore.ParameterSet.Config as cms

topDecayChannelChecker = cms.EDAnalyzer('TopDecayChannelChecker',
  src = cms.InputTag('genParticles'),
  outputFileName = cms.string('TopDecayChannelChecker.root'),
  saveDQMMEs     = cms.bool(True),
  logEvents = cms.uint32(10),                                        
)
