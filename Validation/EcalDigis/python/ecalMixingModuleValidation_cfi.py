import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalSimParameterMap_cff import *
from SimCalorimetry.EcalSimProducers.esElectronicsSim_cff import *
ecalMixingModuleValidation = cms.EDFilter("EcalMixingModuleValidation",
    ecal_sim_parameter_map,
    es_electronics_sim,
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    verbose = cms.untracked.bool(True),
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    ESdigiCollection = cms.InputTag("ecalPreshowerDigis"),
    moduleLabelMC = cms.string('source')
)


