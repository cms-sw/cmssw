import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
#
# special propagator
from TrackingTools.GeomPropagators.BeamHaloPropagatorAlong_cfi import *
from TrackingTools.GeomPropagators.BeamHaloPropagatorOpposite_cfi import *
from TrackingTools.GeomPropagators.BeamHaloPropagatorAny_cfi import *


# clone the steppinghelix propagators
BeamHaloSHPropagatorAlong = SteppingHelixPropagatorAlong.clone(
    ComponentName = 'BeamHaloSHPropagatorAlong'
)
BeamHaloSHPropagatorOpposite = SteppingHelixPropagatorOpposite.clone(
    ComponentName = 'BeamHaloSHPropagatorOpposite'
)
BeamHaloSHPropagatorAny = SteppingHelixPropagatorAny.clone(
    ComponentName = 'BeamHaloSHPropagatorAny'
)
# clone some material propagators
BeamHaloMPropagatorAlong = MaterialPropagator.clone(
    ComponentName = 'BeamHaloMPropagatorAlong',
    MaxDPhi       = 10000,
    useRungeKutta = True
)

BeamHaloMPropagatorOpposite = OppositeMaterialPropagator.clone(
    ComponentName = 'BeamHaloMPropagatorOpposite',
    MaxDPhi       = 10000,
    useRungeKutta = True
)
