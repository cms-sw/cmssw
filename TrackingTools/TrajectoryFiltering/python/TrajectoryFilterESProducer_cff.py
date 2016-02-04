import FWCore.ParameterSet.Config as cms

from TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff import *
from TrackingTools.TrajectoryFiltering.CompositeTrajectoryFilterESProducer_cfi import *
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
ckfBaseTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
ckfBaseTrajectoryFilter.ComponentName = 'ckfBaseTrajectoryFilter' ## actually not needed


