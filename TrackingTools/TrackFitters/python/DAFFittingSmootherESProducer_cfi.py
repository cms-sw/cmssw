import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
DAFFittingSmoother = copy.deepcopy(KFFittingSmoother)
DAFFittingSmoother.ComponentName = 'DAFFittingSmoother'
DAFFittingSmoother.Fitter = 'DAFFitter'
DAFFittingSmoother.Smoother = 'DAFSmoother'

