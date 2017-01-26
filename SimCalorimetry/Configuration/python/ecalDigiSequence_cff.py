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
ecalDigiSequence = cms.Sequence(simEcalTriggerPrimitiveDigis*simEcalDigis*simEcalPreshowerDigis)

from SimCalorimetry.EcalEBTrigPrimProducers.ecalEBTriggerPrimitiveDigis_cff import simEcalEBTriggerPrimitiveDigis as _simEcalEBTriggerPrimitiveDigis
_phase2_ecalDigiSequence = cms.Sequence(_simEcalEBTriggerPrimitiveDigis*ecalDigiSequence)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(ecalDigiSequence,_phase2_ecalDigiSequence)


