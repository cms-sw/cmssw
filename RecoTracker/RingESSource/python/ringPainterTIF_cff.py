import FWCore.ParameterSet.Config as cms

# RingESSource
from RecoTracker.RingMakerESProducer.RingMakerESProducerTIF_cff import *
import copy
from RecoTracker.RingESSource.ringPainter_cfi import *
# RingPainter
ringPainterTIF = copy.deepcopy(ringPainter)
ringPainterTIF.RingLabel = 'TIF'

