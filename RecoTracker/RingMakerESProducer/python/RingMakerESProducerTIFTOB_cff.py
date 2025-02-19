import FWCore.ParameterSet.Config as cms

# geometry
# tracker geometry
# tracker numbering
import copy
from RecoTracker.RingMakerESProducer.RingMakerESProducer_cfi import *
# rings esproducer
ringsTIFTOB = copy.deepcopy(rings)
ringsTIFTOB.Configuration = 'TIFTOB'
ringsTIFTOB.RingAsciiFileName = 'rings_tiftob.dat'
ringsTIFTOB.ComponentName = 'TIFTOB'

