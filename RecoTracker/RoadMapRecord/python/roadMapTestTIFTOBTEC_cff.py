import FWCore.ParameterSet.Config as cms

# roads
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerTIFTOBTEC_cff import *
import copy
from RecoTracker.RoadMapRecord.roadMapTest_cfi import *
# roadMapTest
roadMapTestTIFTOBTEC = copy.deepcopy(roadMapTest)
roadMapTestTIFTOBTEC.RoadLabel = 'TIFTOBTEC'

