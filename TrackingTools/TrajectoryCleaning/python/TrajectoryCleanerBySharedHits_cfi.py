import FWCore.ParameterSet.Config as cms

trajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('TrajectoryCleanerBySharedHits'),
    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
    fractionShared = cms.double(0.19),
    ValidHitBonus = cms.double(5.0),
    MissingHitPenalty = cms.double(20.0),
    allowSharedFirstHit = cms.bool(True)
)
