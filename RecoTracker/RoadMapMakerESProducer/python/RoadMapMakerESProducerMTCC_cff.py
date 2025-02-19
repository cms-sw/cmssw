import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerMTCC_cff import *
import copy
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducer_cfi import *
# RoadMapMaker
roadsMTCC = copy.deepcopy(roads)
roadsMTCC.GeometryStructure = 'MTCC'
roadsMTCC.SeedingType = 'TwoRingSeeds'
roadsMTCC.ComponentName = 'MTCC'
roadsMTCC.RingsLabel = 'MTCC'

