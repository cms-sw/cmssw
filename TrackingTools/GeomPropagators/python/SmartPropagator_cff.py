import FWCore.ParameterSet.Config as cms

# Tracker's propagators 
from TrackingTools.MaterialEffects.RungeKuttaTrackerPropagator_cfi import *
from TrackingTools.MaterialEffects.RungeKuttaTrackerPropagatorOpposite_cfi import *

# Muon's propagators
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *

# Smart propagators

# RungeKuttaTrackerPropagator + SteppingHelixPropagatorAlong (dir = alongMomentum)
from TrackingTools.GeomPropagators.SmartPropagatorRK_cfi import *

# RungeKuttaTrackerPropagatorOpposite + SteppingHelixPropagatorOpposite (dir = oppositeToMomentum)
from TrackingTools.GeomPropagators.SmartPropagatorRKOpposite_cfi import *

# RungeKuttaTrackerPropagator + SteppingHelixPropagatorAny (dir = alongMomentum)
from TrackingTools.GeomPropagators.SmartPropagatorAnyRK_cfi import *

# RungeKuttaTrackerPropagatorOpposite + SteppingHelixPropagatorAny (dir = oppositeToMomentum)
from TrackingTools.GeomPropagators.SmartPropagatorAnyRKOpposite_cfi import *

# PropagatorWithMaterial + SteppingHelixPropagatorAlong (dir = alongMomentum)
from TrackingTools.GeomPropagators.SmartPropagator_cfi import *

# PropagatorWithMaterialOpposite + SteppingHelixPropagatorOpposite (dir = oppositeToMomentum)
from TrackingTools.GeomPropagators.SmartPropagatorOpposite_cfi import *

# PropagatorWithMaterial + SteppingHelixPropagatorAny (dir = alongMomentum)
from TrackingTools.GeomPropagators.SmartPropagatorAny_cfi import *

# PropagatorWithMaterialOpposite + SteppingHelixPropagatorAny (dir = oppositeToMomentum)
from TrackingTools.GeomPropagators.SmartPropagatorAnyOpposite_cfi import *


