import FWCore.ParameterSet.Config as cms

# Rings
from RecoTracker.RingMakerESProducer.RingMakerESProducerMTCC_cff import *
import copy
from RecoTracker.RoadMapESSource.RoadMapESSource_cfi import *
# Roads
roadsMTCC = copy.deepcopy(roads)
roadsMTCC.InputFileName = 'RecoTracker/RoadMapESSource/data/roads_mtcc-0010.dat'
roadsMTCC.ComponentName = 'MTCC'
roadsMTCC.RingsLabel = 'MTCC'

