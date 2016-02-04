import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIFTIBTOB_cff import *
import copy
from RecoTracker.RoadMapESSource.RoadMapESSource_cfi import *
# Roads
roadsTIFTIBTOB = copy.deepcopy(roads)
roadsTIFTIBTOB.InputFileName = 'RecoTracker/RoadMapESSource/data/roads_tiftibtob-0010.dat'
roadsTIFTIBTOB.ComponentName = 'TIFTIBTOB'
roadsTIFTIBTOB.RingsLabel = 'TIFTIBTOB'

