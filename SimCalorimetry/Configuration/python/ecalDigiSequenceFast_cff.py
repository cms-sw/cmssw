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
ecalDigiSequenceFast = cms.Sequence(simEcalTriggerPrimitiveDigis*simEcalDigis*simEcalPreshowerDigis)
from SimGeneral.MixingModule.mixNoPU_cfi import *
mix.digitizers.ecal.doFast = True
#simEcalUnsuppressedDigis.doFast = True


# foo bar baz
# UXZ4qUklRlk40
