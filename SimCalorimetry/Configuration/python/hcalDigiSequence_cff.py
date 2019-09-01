import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisRealistic_cfi import *
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
from SimCalorimetry.HcalTrigPrimProducers.hcalTTPDigis_cfi import *

# simHcalUnsuppressedDigis is now done inside mixing module
hcalDigiTask = cms.Task(simHcalTriggerPrimitiveDigis, simHcalDigis, simHcalTTPDigis)
hcalDigiSequence = cms.Sequence(hcalDigiTask)

# remove HCAL TP sim for premixing stage1
# not needed, sometimes breaks
from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toReplaceWith(hcalDigiTask, cms.Task(simHcalDigis))
