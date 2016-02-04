import FWCore.ParameterSet.Config as cms

# geometry
# tracker geometry
# tracker numbering
import copy
from RecoTracker.RingMakerESProducer.RingMakerESProducer_cfi import *
# rings esproducer
ringsTIFTOBTEC = copy.deepcopy(rings)
ringsTIFTOBTEC.Configuration = 'TIFTOBTEC'
ringsTIFTOBTEC.RingAsciiFileName = 'rings_tiftobtec.dat'
ringsTIFTOBTEC.ComponentName = 'TIFTOBTEC'

