import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIFTIBTOB_cff import *
import copy
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducer_cfi import *
# RoadMapMaker
roadsTIFTIBTOB = copy.deepcopy(roads)
roadsTIFTIBTOB.GeometryStructure = 'TIFTIBTOB'
roadsTIFTIBTOB.SeedingType = 'TwoRingSeeds'
roadsTIFTIBTOB.ComponentName = 'TIFTIBTOB'
roadsTIFTIBTOB.RingsLabel = 'TIFTIBTOB'

