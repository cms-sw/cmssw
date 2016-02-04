import FWCore.ParameterSet.Config as cms

# geometry
# tracker geometry
# tracker numbering
import copy
from RecoTracker.RingESSource.RingESSource_cfi import *
# rings esproducer
ringsTIF = copy.deepcopy(rings)
ringsTIF.InputFileName = 'RecoTracker/RingESSource/data/rings_tif-0004.dat'
ringsTIF.ComponentName = 'TIF'

