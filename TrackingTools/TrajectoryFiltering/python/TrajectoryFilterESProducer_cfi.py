import FWCore.ParameterSet.Config as cms

from TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff import *
trajectoryFilterESProducer = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        CkfBaseTrajectoryFilter_block
    ),
    ComponentName = cms.string('ckfBaseTrajectoryFilter')
)


