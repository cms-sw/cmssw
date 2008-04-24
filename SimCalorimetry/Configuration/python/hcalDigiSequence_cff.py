import FWCore.ParameterSet.Config as cms

#  HCAL digitization
from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import *
from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigis_cfi import *
#  HCAL TPG
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
hcalDigiSequence = cms.Sequence(simHcalUnsuppressedDigis+simHcalTriggerPrimitiveDigis+simHcalDigis)

