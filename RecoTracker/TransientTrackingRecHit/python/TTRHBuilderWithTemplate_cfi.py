import FWCore.ParameterSet.Config as cms

TTRHBuilderAngleAndTemplate = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    ComponentName = cms.string('WithAngleAndTemplate'),
    PixelCPE = cms.string('PixelCPETemplateReco'),
    Matcher = cms.string('StandardMatcher'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
)

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(TTRHBuilderAngleAndTemplate, Phase2StripCPE = cms.string('Phase2StripCPE'))

# from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
# phase1Pixel.toModify(TTRHBuilderAngleAndTemplate, PixelCPE = cms.string('PixelCPEClusterRepair'))

