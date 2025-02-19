import FWCore.ParameterSet.Config as cms

trajectoryCleanerMerger = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('TrajectoryCleanerMerger')
)


