import FWCore.ParameterSet.Config as cms

KFFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    Fitter = cms.string('KFFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('KFSmoother'),
    ComponentName = cms.string('KFFittingSmoother'),
    RejectTracks = cms.bool(True)
)


