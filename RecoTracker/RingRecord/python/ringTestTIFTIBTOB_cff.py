import FWCore.ParameterSet.Config as cms

# ring esproducer
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIFTIBTOB_cff import *
import copy
from RecoTracker.RingRecord.ringTest_cfi import *
# tester
ringTestTIFTIBTOB = copy.deepcopy(ringTest)
ringTestTIFTIBTOB.RingLabel = 'TIFTIBTOB'

