import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import *
from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisRealistic_cfi import *
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
from SimCalorimetry.HcalTrigPrimProducers.hcalTTPDigis_cfi import *

hcalDigiSequence = cms.Sequence(simHcalUnsuppressedDigis
                                +simHcalTriggerPrimitiveDigis
                                +simHcalDigis
                                *simHcalTTPDigis)

