import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIFTOB_cff import *
import copy
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducer_cfi import *
# RoadMapMaker
roadsTIFTOB = copy.deepcopy(roads)
roadsTIFTOB.GeometryStructure = 'TIFTOB'
roadsTIFTOB.SeedingType = 'TwoRingSeeds'
roadsTIFTOB.ComponentName = 'TIFTOB'
roadsTIFTOB.RingsLabel = 'TIFTOB'

