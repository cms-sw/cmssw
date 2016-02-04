import FWCore.ParameterSet.Config as cms

# geometry

# tracker geometry
# tracker numbering
import copy
from RecoTracker.RingESSource.RingESSource_cfi import *
# rings esproducer
ringsMTCC = copy.deepcopy(rings)
ringsMTCC.InputFileName = 'RecoTracker/RingESSource/data/rings_mtcc-0004.dat'
ringsMTCC.ComponentName = 'MTCC'

