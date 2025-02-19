import FWCore.ParameterSet.Config as cms

ttrhbwor = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('WithoutRefit'),
    PixelCPE = cms.string('Fake'),
    Matcher = cms.string('Fake'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False)                          
)


