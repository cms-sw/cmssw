import FWCore.ParameterSet.Config as cms

# roads
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerTIFTIBTOB_cff import *
import copy
from RecoTracker.RoadMapRecord.roadMapTest_cfi import *
# roadMapTest
roadMapTestTIFTIBTOB = copy.deepcopy(roadMapTest)
roadMapTestTIFTIBTOB.RoadLabel = 'TIFTIBTOB'

