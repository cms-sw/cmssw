import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIF_cff import *
import copy
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducer_cfi import *
# RoadMapMaker
roadsTIF = copy.deepcopy(roads)
roadsTIF.GeometryStructure = 'TIF'
roadsTIF.SeedingType = 'TwoRingSeeds'
roadsTIF.ComponentName = 'TIF'
roadsTIF.RingsLabel = 'TIF'

