import FWCore.ParameterSet.Config as cms

# ring esproducer
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIFTOB_cff import *
import copy
from RecoTracker.RingRecord.ringTest_cfi import *
# tester
ringTestTIFTOB = copy.deepcopy(ringTest)
ringTestTIFTOB.RingLabel = 'TIFTOB'

