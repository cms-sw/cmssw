import FWCore.ParameterSet.Config as cms

# RoadMapESSource
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerTIF_cff import *
import copy
from RecoTracker.RoadMapESSource.roadPainter_cfi import *
# RoadPainter
roadPainterTIF = copy.deepcopy(roadPainter)
roadPainterTIF.RoadLabel = 'TIF'
roadPainterTIF.RingLabel = 'TIF'

