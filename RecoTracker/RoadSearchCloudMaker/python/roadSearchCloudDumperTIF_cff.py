import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIF_cff import *
import copy
from RecoTracker.RoadSearchCloudMaker.roadSearchCloudDumper_cfi import *
# include RoadSearchCloudDumper
roadSearchCloudDumperTIF = copy.deepcopy(roadSearchCloudDumper)
roadSearchCloudDumperTIF.RoadSearchCloudInputTag = 'roadSearchCloudsTIF'
roadSearchCloudDumperTIF.RingsLabel = 'TIF'

