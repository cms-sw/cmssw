import FWCore.ParameterSet.Config as cms

# RoadMapESSource
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerTIFTIBTOB_cff import *
import copy
from RecoTracker.RoadMapESSource.roadPainter_cfi import *
# RoadPainter
roadPainterTIFTIBTOB = copy.deepcopy(roadPainter)
roadPainterTIFTIBTOB.RoadLabel = 'TIFTIBTOB'
roadPainterTIFTIBTOB.RingLabel = 'TIFTIBTOB'

