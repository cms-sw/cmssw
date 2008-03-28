import FWCore.ParameterSet.Config as cms

# ring esproducer
from RecoTracker.RingMakerESProducer.RingMakerESProducerMTCC_cff import *
import copy
from RecoTracker.RingRecord.ringTest_cfi import *
# tester
ringTestMTCC = copy.deepcopy(ringTest)
ringTestMTCC.RingLabel = 'MTCC'

