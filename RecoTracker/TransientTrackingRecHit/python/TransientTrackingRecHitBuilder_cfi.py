import FWCore.ParameterSet.Config as cms

ttrhbwr = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    ComponentName = cms.string('WithTrackAngle'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)


