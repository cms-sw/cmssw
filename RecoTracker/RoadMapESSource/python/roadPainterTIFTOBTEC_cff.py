import FWCore.ParameterSet.Config as cms

# RoadMapESSource
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerTIFTOBTEC_cff import *
import copy
from RecoTracker.RoadMapESSource.roadPainter_cfi import *
# RoadPainter
roadPainterTIFTOBTEC = copy.deepcopy(roadPainter)
roadPainterTIFTOBTEC.RoadLabel = 'TIFTOBTEC'
roadPainterTIFTOBTEC.RingLabel = 'TIFTOBTEC'

