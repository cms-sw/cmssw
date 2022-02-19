import FWCore.ParameterSet.Config as cms

ttrhbwr = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    Phase2StripCPE = cms.string(''),
    ComponentName = cms.string('WithTrackAngle'),
    PixelCPE = cms.string('PixelCPEGeneric'),
    Matcher = cms.string('StandardMatcher'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
)

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(ttrhbwr, 
                             Phase2StripCPE = 'Phase2StripCPE',
                             StripCPE = 'FakeStripCPE')

from Configuration.Eras.Modifier_phase2_brickedPixels_cff import phase2_brickedPixels
phase2_brickedPixels.toModify(ttrhbwr, PixelCPE = 'PixelCPEGenericForBricked')

