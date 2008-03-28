import FWCore.ParameterSet.Config as cms

# RoadMapESSource
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerTIFTOB_cff import *
import copy
from RecoTracker.RoadMapESSource.roadPainter_cfi import *
# RoadPainter
roadPainterTIFTOB = copy.deepcopy(roadPainter)
roadPainterTIFTOB.RoadLabel = 'TIFTOB'
roadPainterTIFTOB.RingLabel = 'TIFTOB'

