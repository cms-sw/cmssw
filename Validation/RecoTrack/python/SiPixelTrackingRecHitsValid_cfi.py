import FWCore.ParameterSet.Config as cms

PixelTrackingRecHitsValid = cms.EDFilter("SiPixelTrackingRecHitsValid",
    src = cms.untracked.string('generalTracks'),
    outputFile = cms.untracked.string('pixeltrackingrechitshist.root'),
    Fitter = cms.string('KFFittingSmoother'),
    # do we check that the simHit associated with recHit is of the expected particle type ?
    checkType = cms.bool(True),
    MTCCtrack = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    # the type of particle that the simHit associated with recHits should be
    genType = cms.int32(13),
    Propagator = cms.string('PropagatorWithMaterial')
)


