import FWCore.ParameterSet.Config as cms

# ring esproducer
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIFTOBTEC_cff import *
import copy
from RecoTracker.RingRecord.ringTest_cfi import *
# tester
ringTestTIFTOBTEC = copy.deepcopy(ringTest)
ringTestTIFTOBTEC.RingLabel = 'TIFTOBTEC'

