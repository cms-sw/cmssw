import FWCore.ParameterSet.Config as cms

# RingESSource
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIFTOB_cff import *
import copy
from RecoTracker.RingESSource.ringPainter_cfi import *
# RingPainter
ringPainterTIFTOB = copy.deepcopy(ringPainter)
ringPainterTIFTOB.RingLabel = 'TIFTOB'

