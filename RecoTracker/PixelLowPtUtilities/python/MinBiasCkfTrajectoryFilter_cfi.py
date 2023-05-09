import FWCore.ParameterSet.Config as cms

# Trajectory filter for min bias
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
ckfBaseTrajectoryFilterForMinBias = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt               = 0.075
)
from RecoTracker.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
from RecoTracker.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import *

# Composite filter
MinBiasCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters   = [cms.PSet(refToPSet_ = cms.string('ckfBaseTrajectoryFilterForMinBias')),
                 cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)

