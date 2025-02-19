import FWCore.ParameterSet.Config as cms

# roads
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerTIF_cff import *
import copy
from RecoTracker.RoadMapRecord.roadMapTest_cfi import *
# roadMapTest
roadMapTestTIF = copy.deepcopy(roadMapTest)
roadMapTestTIF.RoadLabel = 'TIF'

