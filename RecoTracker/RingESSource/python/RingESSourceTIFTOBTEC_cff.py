import FWCore.ParameterSet.Config as cms

# geometry
# tracker geometry
# tracker numbering
import copy
from RecoTracker.RingESSource.RingESSource_cfi import *
# rings esproducer
ringsTIFTOBTEC = copy.deepcopy(rings)
ringsTIFTOBTEC.InputFileName = 'RecoTracker/RingESSource/data/rings_tiftobtec-0004.dat'
ringsTIFTOBTEC.ComponentName = 'TIFTOBTEC'

