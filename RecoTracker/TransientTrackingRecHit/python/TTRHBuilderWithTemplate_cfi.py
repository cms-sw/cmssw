import FWCore.ParameterSet.Config as cms

from RecoTracker.TransientTrackingRecHit.tkTransientTrackingRecHitBuilderESProducer_cfi import tkTransientTrackingRecHitBuilderESProducer
TTRHBuilderAngleAndTemplate = tkTransientTrackingRecHitBuilderESProducer.clone(StripCPE = 'StripCPEfromTrackAngle',
                                                                               Phase2StripCPE = '',
                                                                               ComponentName = 'WithAngleAndTemplate',
                                                                               PixelCPE = 'PixelCPETemplateReco',
                                                                               Matcher = 'StandardMatcher',
                                                                               ComputeCoarseLocalPositionFromDisk = False)

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(TTRHBuilderAngleAndTemplate, 
                             Phase2StripCPE = 'Phase2StripCPE',
                             StripCPE = 'FakeStripCPE')

# uncomment these two lines to turn on Cluster Repair CPE
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(TTRHBuilderAngleAndTemplate, PixelCPE = 'PixelCPEClusterRepair')

# Turn off template reco for phase 2 (when not supported)
from Configuration.ProcessModifiers.PixelCPEGeneric_cff import PixelCPEGeneric
PixelCPEGeneric.toModify(TTRHBuilderAngleAndTemplate, PixelCPE = 'PixelCPEGeneric')

