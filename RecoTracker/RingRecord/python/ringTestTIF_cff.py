import FWCore.ParameterSet.Config as cms

# ring esproducer
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIF_cff import *
import copy
from RecoTracker.RingRecord.ringTest_cfi import *
# tester
ringTestTIF = copy.deepcopy(ringTest)
ringTestTIF.RingLabel = 'TIF'

