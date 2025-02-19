import FWCore.ParameterSet.Config as cms

# geometry

# tracker geometry
# tracker numbering
import copy
from RecoTracker.RingMakerESProducer.RingMakerESProducer_cfi import *
# rings esproducer
ringsMTCC = copy.deepcopy(rings)
ringsMTCC.RingAsciiFileName = 'rings_mtcc.dat'
ringsMTCC.ComponentName = 'MTCC'

