import FWCore.ParameterSet.Config as cms

trajectoryCleanerBySharedSeeds = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('TrajectoryCleanerBySharedSeeds'),
    ComponentType = cms.string('TrajectoryCleanerBySharedSeeds')
)


