import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIF_cff import *
import copy
from RecoTracker.RoadMapESSource.RoadMapESSource_cfi import *
# Roads
roadsTIF = copy.deepcopy(roads)
roadsTIF.InputFileName = 'RecoTracker/RoadMapESSource/data/roads_tif-0010.dat'
roadsTIF.ComponentName = 'TIF'
roadsTIF.RingsLabel = 'TIF'

