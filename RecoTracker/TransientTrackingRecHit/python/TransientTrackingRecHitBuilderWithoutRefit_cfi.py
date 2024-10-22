import FWCore.ParameterSet.Config as cms

from RecoTracker.TransientTrackingRecHit.tkTransientTrackingRecHitBuilderESProducer_cfi import tkTransientTrackingRecHitBuilderESProducer
ttrhbwor =  tkTransientTrackingRecHitBuilderESProducer.clone(StripCPE = 'Fake',
                                                             Phase2StripCPE = '',
                                                             ComponentName = 'WithoutRefit',
                                                             PixelCPE = 'Fake',
                                                             Matcher = 'Fake',
                                                             ComputeCoarseLocalPositionFromDisk = False)

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toModify(ttrhbwor, 
                             Phase2StripCPE = 'Phase2StripCPE',
                             StripCPE = 'FakeStripCPE')

