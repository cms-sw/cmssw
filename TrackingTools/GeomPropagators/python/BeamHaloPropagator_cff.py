import FWCore.ParameterSet.Config as cms

import copy
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
# clone the steppinghelix propagators
BeamHaloSHPropagatorAlong = copy.deepcopy(SteppingHelixPropagatorAlong)
import copy
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
BeamHaloSHPropagatorOpposite = copy.deepcopy(SteppingHelixPropagatorOpposite)
import copy
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
BeamHaloSHPropagatorAny = copy.deepcopy(SteppingHelixPropagatorAny)
import copy
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# clone some material propagators
BeamHaloMPropagatorAlong = copy.deepcopy(MaterialPropagator)
import copy
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
BeamHaloMPropagatorOpposite = copy.deepcopy(OppositeMaterialPropagator)
#
# special propagator
from TrackingTools.GeomPropagators.BeamHaloPropagatorAlong_cfi import *
from TrackingTools.GeomPropagators.BeamHaloPropagatorOpposite_cfi import *
from TrackingTools.GeomPropagators.BeamHaloPropagatorAny_cfi import *
BeamHaloSHPropagatorAlong.ComponentName = 'BeamHaloSHPropagatorAlong'
BeamHaloSHPropagatorOpposite.ComponentName = 'BeamHaloSHPropagatorOpposite'
BeamHaloSHPropagatorAny.ComponentName = 'BeamHaloSHPropagatorAny'
BeamHaloMPropagatorAlong.ComponentName = 'BeamHaloMPropagatorAlong'
BeamHaloMPropagatorAlong.MaxDPhi = 10000
BeamHaloMPropagatorOpposite.ComponentName = 'BeamHaloMPropagatorOpposite'
BeamHaloMPropagatorOpposite.MaxDPhi = 10000

BeamHaloMPropagatorAlong.useRungeKutta = True
BeamHaloMPropagatorOpposite.useRungeKutta = True
