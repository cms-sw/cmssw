import FWCore.ParameterSet.Config as cms

TTRHBuilderAngleAndTemplate = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    ComponentName = cms.string('WithAngleAndTemplate'),
    PixelCPE = cms.string('PixelCPETemplateReco'),
    Matcher = cms.string('StandardMatcher'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
)

from Configuration.StandardSequences.Eras import eras
eras.trackingPhase2PU140.toModify(TTRHBuilderAngleAndTemplate, Phase2StripCPE = cms.string('Phase2StripCPEGeometric'))

