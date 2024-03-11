import FWCore.ParameterSet.Config as cms

#--- HCAL Trigger Primitive generation ---#
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
hcalTPDigis = simHcalTriggerPrimitiveDigis.clone()
hcalTPDigis.inputLabel = cms.VInputTag('simHcalDigis','simHcalDigis')
#--- The following settings should be used on data only ---#
# hcalTPDigis.FrontEndFormatError = cms.bool(True)
# HcalTPGCoderULUT.LUTGenerationMode = cms.bool(False) 

from SimCalorimetry.HcalTrigPrimProducers.hcalTTPDigis_cfi import *
from SimCalorimetry.HcalTrigPrimProducers.hcalTTPRecord_cfi import * 

hcalTTPSequence = cms.Sequence( simHcalTTPDigis + simHcalTTPRecord ) 

# foo bar baz
