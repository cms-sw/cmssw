import FWCore.ParameterSet.Config as cms

from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
import copy
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
RKTrackerPropagator = copy.deepcopy(MaterialPropagator)
# Muon's propagators
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
# Tracker's propagators 
from TrackingTools.GeomPropagators.SmartPropagator_cfi import *
from TrackingTools.GeomPropagators.SmartPropagatorRK_cfi import *
from TrackingTools.GeomPropagators.SmartPropagatorOpposite_cfi import *
from TrackingTools.GeomPropagators.SmartPropagatorAny_cfi import *
from TrackingTools.GeomPropagators.SmartPropagatorAnyRK_cfi import *
from TrackingTools.GeomPropagators.SmartPropagatorAnyOpposite_cfi import *
RKTrackerPropagator.ComponentName = 'RKTrackerPropagator'
RKTrackerPropagator.useRungeKutta = True

