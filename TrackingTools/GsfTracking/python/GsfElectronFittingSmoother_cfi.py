import FWCore.ParameterSet.Config as cms

GsfElectronFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    Fitter = cms.string('GsfTrajectoryFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('GsfTrajectorySmoother'),
    ComponentName = cms.string('GsfElectronFittingSmoother'),
    RejectTracks = cms.bool(True)
)


