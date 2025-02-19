import FWCore.ParameterSet.Config as cms

# RoadMapESSource
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerMTCC_cff import *
import copy
from RecoTracker.RoadMapESSource.roadPainter_cfi import *
# RoadPainter
roadPainterMTCC = copy.deepcopy(roadPainter)
roadPainterMTCC.RoadLabel = 'MTCC'
roadPainterMTCC.RingLabel = 'MTCC'

