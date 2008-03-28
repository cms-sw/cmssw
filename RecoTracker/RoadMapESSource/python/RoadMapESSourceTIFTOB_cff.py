import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIFTOB_cff import *
import copy
from RecoTracker.RoadMapESSource.RoadMapESSource_cfi import *
# Roads
roadsTIFTOB = copy.deepcopy(roads)
roadsTIFTOB.InputFileName = 'RecoTracker/RoadMapESSource/data/roads_tiftob-0010.dat'
roadsTIFTOB.ComponentName = 'TIFTOB'
roadsTIFTOB.RingsLabel = 'TIFTOB'

