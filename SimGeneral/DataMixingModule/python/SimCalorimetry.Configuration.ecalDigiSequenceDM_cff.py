import FWCore.ParameterSet.Config as cms

# unsuppressed digis simulation - fast preshower
from SimCalorimetry.EcalSimProducers.ecaldigi_cfi import *
# ECAL Trigger Primitives (needed by SRP)
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
# Selective Readout Processor producer
from SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi import *
# Preshower Zero suppression producer
from SimCalorimetry.EcalZeroSuppressionProducers.ecalPreshowerDigis_cfi import *
ecalDigiSequence = cms.Sequence(simEcalTriggerPrimitiveDigis*simEcalDigis)
simEcalUnsuppressedDigis.doFast = True
simEcalPreshowerDigis.ESNoiseSigma = 2.98595

# Re-define inputs to point at DataMixer output
simEcalTriggerPrimitiveDigis.Label = cms.string('mixData')
simEcalTriggerPrimitiveDigis.InstanceEB = cms.string('EBDigiCollectionDM')
simEcalTriggerPrimitiveDigis.InstanceEE = cms.string('EEDigiCollectionDM')
#
simEcalDigis.digiProducer = cms.string('mixData')
simEcalDigis.EBdigiCollection = cms.string('EBDigiCollectionDM')
simEcalDigis.EEdigiCollection = cms.string('EEDigiCollectionDM')


