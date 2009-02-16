import FWCore.ParameterSet.Config as cms

print "--------------------------------"
print "------------- ATTENTION --------"
print "--------------------------------"
print "Please do not use this import anymore."
print "Please use RecoTracker.TrackProducer.TrackProducer_cfi instead"
print "--------------------------------"
print "--------------------------------"

import RecoTracker.TrackProducer.TrackProducer_cfi
ctfWithMaterialTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone()

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



