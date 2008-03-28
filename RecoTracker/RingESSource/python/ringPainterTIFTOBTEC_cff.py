import FWCore.ParameterSet.Config as cms

#ring RingESSource
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIFTOBTEC_cff import *
import copy
from RecoTracker.RingESSource.ringPainter_cfi import *
# RingPainter
ringPainterTIFTOBTEC = copy.deepcopy(ringPainter)
ringPainterTIFTOBTEC.RingLabel = 'TIFTOBTEC'

