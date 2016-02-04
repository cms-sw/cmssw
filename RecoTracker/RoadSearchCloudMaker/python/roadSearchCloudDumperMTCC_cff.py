import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerMTCC_cff import *
import copy
from RecoTracker.RoadSearchCloudMaker.roadSearchCloudDumper_cfi import *
# include RoadSearchCloudDumper
roadSearchCloudDumperMTCC = copy.deepcopy(roadSearchCloudDumper)
roadSearchCloudDumperMTCC.RoadSearchCloudInputTag = 'roadSearchCloudsMTCC'
roadSearchCloudDumperMTCC.RingsLabel = 'MTCC'

