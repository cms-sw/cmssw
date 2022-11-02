import FWCore.ParameterSet.Config as cms

from RecoTracker.TransientTrackingRecHit.tkTransientTrackingRecHitBuilderESProducer_cfi import tkTransientTrackingRecHitBuilderESProducer
ttrhbwr =  tkTransientTrackingRecHitBuilderESProducer.clone(StripCPE = 'StripCPEfromTrackAngle',
                                                            Phase2StripCPE = '',
                                                            ComponentName = 'WithTrackAngle',
                                                            PixelCPE = 'PixelCPEGeneric',
                                                            Matcher = 'StandardMatcher',
                                                            ComputeCoarseLocalPositionFromDisk = False)

TTRHBuilderFast = ttrhbwr.clone(ComponentName = 'WithoutAngleFast',
                                PixelCPE = 'PixelCPEFast')

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(ttrhbwr, 
                             Phase2StripCPE = 'Phase2StripCPE',
                             StripCPE = 'FakeStripCPE')

