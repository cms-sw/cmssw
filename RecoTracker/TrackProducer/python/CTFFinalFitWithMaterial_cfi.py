import FWCore.ParameterSet.Config as cms

import RecoTracker.TrackProducer.TrackProducer_cfi
ctfWithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK')
    )

#ctfWithMaterialTracks = cms.EDProducer("TrackProducer",
#    src = cms.InputTag("ckfTrackCandidates"),
#    clusterRemovalInfo = cms.InputTag(""),
#    beamSpot = cms.InputTag("offlineBeamSpot"),
#    Fitter = cms.string('KFFittingSmootherWithOutliersRejectionAndRK'),
#    useHitsSplitting = cms.bool(False),
#    alias = cms.untracked.string('ctfWithMaterialTracks'),
#    TrajectoryInEvent = cms.bool(True),
#    TTRHBuilder = cms.string('WithAngleAndTemplate'),
#    AlgorithmName = cms.string('undefAlgorithm'),
#    Propagator = cms.string('RungeKuttaTrackerPropagator')
#)



