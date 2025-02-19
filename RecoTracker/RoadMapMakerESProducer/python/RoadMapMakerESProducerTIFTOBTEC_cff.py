import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIFTOBTEC_cff import *
import copy
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducer_cfi import *
# RoadMapMaker
roadsTIFTOBTEC = copy.deepcopy(roads)
roadsTIFTOBTEC.GeometryStructure = 'TIFTOBTEC'
roadsTIFTOBTEC.SeedingType = 'TwoRingSeeds'
roadsTIFTOBTEC.ComponentName = 'TIFTOBTEC'
roadsTIFTOBTEC.RingsLabel = 'TIFTOBTEC'

