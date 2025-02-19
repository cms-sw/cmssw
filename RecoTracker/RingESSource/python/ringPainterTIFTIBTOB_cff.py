import FWCore.ParameterSet.Config as cms

# RingESSource
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIFTIBTOB_cff import *
import copy
from RecoTracker.RingESSource.ringPainter_cfi import *
# RingPainter
ringPainterTIFTIBTOB = copy.deepcopy(ringPainter)
ringPainterTIFTIBTOB.RingLabel = 'TIFTIBTOB'

