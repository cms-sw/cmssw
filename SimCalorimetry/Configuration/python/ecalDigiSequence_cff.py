import FWCore.ParameterSet.Config as cms

# unsuppressed digis simulation - fast preshower
from SimCalorimetry.EcalSimProducers.ecaldigi_cfi import *
# ECAL Trigger Primitives (needed by SRP)
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
# Selective Readout Processor producer
from SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi import *
# Preshower Zero suppression producer
from SimCalorimetry.EcalZeroSuppressionProducers.ecalPreshowerDigis_cfi import *
# simEcalUnsuppressedDigis is now done inside mixing module
ecalDigiTask = cms.Task(simEcalTriggerPrimitiveDigis, simEcalDigis, simEcalPreshowerDigis)
ecalDigiSequence = cms.Sequence(ecalDigiTask)


# This is extra, since the configuration skips it anyway.  Belts and suspenders.
from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toReplaceWith(ecalDigiTask, ecalDigiTask.copyAndExclude([simEcalPreshowerDigis]))

from SimCalorimetry.EcalEBTrigPrimProducers.ecalEBTriggerPrimitiveDigis_cff import *
_phase2_ecalDigiTask = ecalDigiTask.copy()
_phase2_ecalDigiTask.add(simEcalEBTriggerPrimitiveDigis)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(ecalDigiTask,_phase2_ecalDigiTask)


