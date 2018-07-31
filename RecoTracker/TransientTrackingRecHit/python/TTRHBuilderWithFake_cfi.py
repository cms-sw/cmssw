import FWCore.ParameterSet.Config as cms

pixelFake = cms.ESProducer("FakePixelCPEESProducer",
    ComponentName = cms.string('FakePixelCPE')
)

from RecoLocalTracker.SiStripRecHitConverter.StripCPEESProducer_cfi import *
StripCPEfromFake = stripCPEESProducer.clone()
StripCPEfromFake.ComponentName = 'FakeStripCPE'
StripCPEfromFake.ComponentType = 'FakeStripCPE'


TTRHBuilderFake = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('FakeStripCPE'),
    ComponentName = cms.string('Fake'),
    PixelCPE = cms.string('FakePixelCPE'),
    Matcher = cms.string('StandardMatcher'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
)

