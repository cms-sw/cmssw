import FWCore.ParameterSet.Config as cms

# geometry
# tracker geometry
# tracker numbering
import copy
from RecoTracker.RingMakerESProducer.RingMakerESProducer_cfi import *
# rings esproducer
ringsTIFTIBTOB = copy.deepcopy(rings)
ringsTIFTIBTOB.Configuration = 'TIFTIBTOB'
ringsTIFTIBTOB.RingAsciiFileName = 'rings_tiftibtob.dat'
ringsTIFTIBTOB.ComponentName = 'TIFTIBTOB'

