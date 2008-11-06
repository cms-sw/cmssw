import FWCore.ParameterSet.Config as cms

ctfWithMaterialTracksMTF = cms.EDProducer("MTFTrackProducer",
    src = cms.InputTag("ckfTrackCandidates"),
    UpdatorName = cms.string('SiTrackerMultiRecHitUpdatorMTF'),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('MRHFittingSmoother'),
    MeasurementCollector = cms.string('simpleMTFHitCollector'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    MinHits = cms.int32(3)
)


