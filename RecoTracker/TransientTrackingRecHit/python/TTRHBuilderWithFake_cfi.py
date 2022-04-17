import FWCore.ParameterSet.Config as cms

pixelFake = cms.ESProducer("FakePixelCPEESProducer",
    ComponentName = cms.string('FakePixelCPE')
)

from RecoLocalTracker.SiStripRecHitConverter.StripCPEESProducer_cfi import *
StripCPEfromFake = stripCPEESProducer.clone(
    ComponentName = 'FakeStripCPE',
    ComponentType = 'FakeStripCPE'
)

from RecoTracker.TransientTrackingRecHit.tkTransientTrackingRecHitBuilderESProducer_cfi import tkTransientTrackingRecHitBuilderESProducer
TTRHBuilderFake = tkTransientTrackingRecHitBuilderESProducer.clone(StripCPE = 'FakeStripCPE',
                                                                   ComponentName = 'Fake',
                                                                   PixelCPE = 'FakePixelCPE',
                                                                   Matcher = 'StandardMatcher',
                                                                   ComputeCoarseLocalPositionFromDisk = False)

