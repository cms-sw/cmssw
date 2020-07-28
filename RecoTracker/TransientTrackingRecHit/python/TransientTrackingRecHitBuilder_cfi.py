import FWCore.ParameterSet.Config as cms

ttrhbwr = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    ComponentName = cms.string('WithTrackAngle'),
    PixelCPE = cms.string('PixelCPEGeneric'),
    Matcher = cms.string('StandardMatcher'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
)

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(ttrhbwr, Phase2StripCPE = cms.string('Phase2StripCPE'))

