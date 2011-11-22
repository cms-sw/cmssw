import FWCore.ParameterSet.Config as cms

simHcalUpgradeTriggerPrimitiveDigis = cms.EDProducer('HcalUpgradeTrigPrimDigiProducer',
  hbheDigis          = cms.InputTag("simHcalDigis"),
  hfDigis            = cms.InputTag("simHcalDigis"),
  latency            = cms.int32(1),
  weights            = cms.vdouble(1.0, 1.0), ##hardware algo 
  peakFinder         = cms.bool(True),
  FGThreshold        = cms.int32 (12),
  ZSThreshold        = cms.int32 (1 ),
  MinSignalThreshold = cms.int32(0),
  PMTNoiseThreshold  = cms.int32(0),                      
  NumberOfSamples    = cms.int32(4),
  NumberOfPresamples = cms.int32(2),
  RunZS              = cms.untracked.bool ( False ),                                                   
  excludeDepth5 = cms.bool(True)
)
