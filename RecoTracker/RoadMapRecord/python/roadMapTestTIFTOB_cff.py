import FWCore.ParameterSet.Config as cms

# roads
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerTIFTOB_cff import *
import copy
from RecoTracker.RoadMapRecord.roadMapTest_cfi import *
# roadMapTest
roadMapTestTIFTOB = copy.deepcopy(roadMapTest)
roadMapTestTIFTOB.RoadLabel = 'TIFTOB'

