import FWCore.ParameterSet.Config as cms

KFFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",

    EstimateCut = cms.double(-1.0),

    # ggiurgiu@fnal.gov : Any value lower than -15 turns off this cut.
    # Recommended default value: -14.0. This will reject only the worst hits with negligible loss in track efficiency.  
    LogPixelProbabilityCut = cms.double(-16.0),                               

    Fitter = cms.string('KFFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('KFSmoother'),
    ComponentName = cms.string('KFFittingSmoother'),
    RejectTracks = cms.bool(True),
    BreakTrajWith2ConsecutiveMissing = cms.bool(True),
    NoInvalidHitsBeginEnd  = cms.bool(True)
)


