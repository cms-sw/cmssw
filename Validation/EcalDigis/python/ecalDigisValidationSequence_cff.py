import FWCore.ParameterSet.Config as cms

# Run the complete ECAL digis validation set
from Validation.EcalDigis.ecalDigisValidation_cfi import *
from Validation.EcalDigis.ecalBarrelDigisValidation_cfi import *
from Validation.EcalDigis.ecalEndcapDigisValidation_cfi import *
from Validation.EcalDigis.ecalPreshowerDigisValidation_cfi import *
from Validation.EcalDigis.ecalSelectiveReadoutValidation_cfi import *
ecalDigisValidationSequence = cms.Sequence(ecalDigisValidation*ecalBarrelDigisValidation*ecalEndcapDigisValidation*ecalPreshowerDigisValidation*ecalSelectiveReadoutValidation)

from Validation.EcalDigis.ecalDigisValidationPh2_cfi import *
ecalDigisValidationSequencePhase2 = cms.Sequence(ecalDigisValidationPh2)

from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
def _modifyEcalForPh2(process):
    process.load("SimCalorimetry.EcalSimProducers.esCATIAGainProducer_cfi")
modifyVal_Phase2Ecal = phase2_ecal_devel.makeProcessModifier(_modifyEcalForPh2)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
(phase2_ecal_devel & premix_stage2).toModify(ecalDigisValidationPh2, digiCollection = 'mixData')
