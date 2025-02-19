import FWCore.ParameterSet.Config as cms

# geometry
# tracker geometry
# tracker numbering
import copy
from RecoTracker.RingMakerESProducer.RingMakerESProducer_cfi import *
# rings esproducer
ringsTIF = copy.deepcopy(rings)
ringsTIF.Configuration = 'TIF'
ringsTIF.RingAsciiFileName = 'rings_tif.dat'
ringsTIF.ComponentName = 'TIF'

