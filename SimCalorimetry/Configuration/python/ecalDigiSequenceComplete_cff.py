import FWCore.ParameterSet.Config as cms

# unsuppressed digis simulation - full ES digitization
from SimCalorimetry.EcalSimProducers.ecaldigi_cfi import *
# ECAL Trigger Primitives (needed by SRP)
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
# Selective Readout Processor producer
from SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cff import *
# Preshower Zero suppression producer
from SimCalorimetry.EcalZeroSuppressionProducers.ecalPreshowerDigis_cfi import *
ecalDigiSequenceComplete = cms.Sequence(simEcalUnsuppressedDigis*simEcalTriggerPrimitiveDigis*simEcalDigis*simEcalPreshowerDigis)


