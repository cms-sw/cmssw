import FWCore.ParameterSet.Config as cms

ttrhbwor = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('WithoutRefit'),
    PixelCPE = cms.string('Fake'),
    Matcher = cms.string('Fake'),
    ComputeCoarseLocalPositionFromDisk = cms.bool(False),
)

from Configuration.StandardSequences.Eras import eras
eras.trackingPhase2PU140.toModify(ttrhbwor, Phase2StripCPE = cms.string('Phase2StripCPEGeometric'))

