import FWCore.ParameterSet.Config as cms

# geometry
# tracker geometry
# tracker numbering
import copy
from RecoTracker.RingMakerESProducer.RingMakerESProducer_cfi import *
# rings esproducer
ringsP5 = copy.deepcopy(rings)
ringsP5.Configuration = 'P5'
ringsP5.RingAsciiFileName = 'rings_p5.dat'
ringsP5.ComponentName = 'P5'

