import FWCore.ParameterSet.Config as cms

# RingESSource
from RecoTracker.RingMakerESProducer.RingMakerESProducerMTCC_cff import *
import copy
from RecoTracker.RingESSource.ringPainter_cfi import *
# RingPainter
ringPainterMTCC = copy.deepcopy(ringPainter)
ringPainterMTCC.RingLabel = 'MTCC'

