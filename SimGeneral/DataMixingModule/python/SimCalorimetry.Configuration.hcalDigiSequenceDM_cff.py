import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import *
from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisRealistic_cfi import *
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
#
# Redefine inputs to point at DataMixer output:
#
simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(cms.InputTag('mixData'),cms.InputTag('mixData'))
simHcalDigis.digiLabel = cms.InputTag("mixData")

hcalDigiSequence = cms.Sequence(simHcalTriggerPrimitiveDigis+simHcalDigis)

