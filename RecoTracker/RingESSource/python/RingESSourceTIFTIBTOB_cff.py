import FWCore.ParameterSet.Config as cms

# geometry
# tracker geometry
# tracker numbering
import copy
from RecoTracker.RingESSource.RingESSource_cfi import *
# rings esproducer
ringsTIFTIBTOB = copy.deepcopy(rings)
ringsTIFTIBTOB.InputFileName = 'RecoTracker/RingESSource/data/rings_tiftibtob-0004.dat'
ringsTIFTIBTOB.ComponentName = 'TIFTIBTOB'

