import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerP5_cff import *
import copy
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducer_cfi import *
# RoadMapMaker
roadsP5 = copy.deepcopy(roads)
roadsP5.GeometryStructure = 'P5'
roadsP5.SeedingType = 'TwoRingSeeds'
roadsP5.ComponentName = 'P5'
roadsP5.RingsLabel = 'P5'

