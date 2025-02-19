import FWCore.ParameterSet.Config as cms

# roads
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerMTCC_cff import *
import copy
from RecoTracker.RoadMapRecord.roadMapTest_cfi import *
# roadMapTest
roadMapTestMTCC = copy.deepcopy(roadMapTest)
roadMapTestMTCC.RoadLabel = 'MTCC'

