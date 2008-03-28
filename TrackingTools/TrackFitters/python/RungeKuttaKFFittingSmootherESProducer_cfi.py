import FWCore.ParameterSet.Config as cms

RKFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    Fitter = cms.string('RKFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('RKSmoother'),
    ComponentName = cms.string('RKFittingSmoother'),
    RejectTracks = cms.bool(True)
)


