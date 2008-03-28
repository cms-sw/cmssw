import FWCore.ParameterSet.Config as cms

from TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff import *
from TrackingTools.TrajectoryFiltering.CompositeTrajectoryFilterESProducer_cfi import *
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
ckfBaseTrajectoryFilter = copy.deepcopy(trajectoryFilterESProducer)
ckfBaseTrajectoryFilter.ComponentName = 'ckfBaseTrajectoryFilter' ## actually not needed


