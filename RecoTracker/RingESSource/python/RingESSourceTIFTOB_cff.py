import FWCore.ParameterSet.Config as cms

# geometry
# tracker geometry
# tracker numbering
import copy
from RecoTracker.RingESSource.RingESSource_cfi import *
# rings esproducer
ringsTIFTOB = copy.deepcopy(rings)
ringsTIFTOB.InputFileName = 'RecoTracker/RingESSource/data/rings_tiftob-0004.dat'
ringsTIFTOB.ComponentName = 'TIFTOB'

