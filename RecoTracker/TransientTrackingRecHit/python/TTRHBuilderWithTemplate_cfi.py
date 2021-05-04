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

# uncomment these two lines to turn on Cluster Repair CPE
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(TTRHBuilderAngleAndTemplate, PixelCPE = 'PixelCPEClusterRepair')

# Turn off template reco for phase 2 (when not supported)
from Configuration.ProcessModifiers.PixelCPEGeneric_cff import PixelCPEGeneric
PixelCPEGeneric.toModify(TTRHBuilderAngleAndTemplate, PixelCPE = 'PixelCPEGeneric')
