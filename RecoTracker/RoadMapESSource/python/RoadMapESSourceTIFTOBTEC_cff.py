import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIFTOBTEC_cff import *
import copy
from RecoTracker.RoadMapESSource.RoadMapESSource_cfi import *
# Roads
roadsTIFTOBTEC = copy.deepcopy(roads)
roadsTIFTOBTEC.InputFileName = 'RecoTracker/RoadMapESSource/data/roads_tiftobtec-0010.dat'
roadsTIFTOBTEC.ComponentName = 'TIFTOBTEC'
roadsTIFTOBTEC.RingsLabel = 'TIFTOBTEC'

