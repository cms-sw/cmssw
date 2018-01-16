import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalSimParameterMap_cff import *
from SimCalorimetry.EcalSimProducers.esElectronicsSim_cff import *
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalMixingModuleValidation = DQMEDAnalyzer('EcalMixingModuleValidation',
    ecal_sim_parameter_map,
    es_electronics_sim,
    hitsProducer = cms.string('g4SimHits'),
    EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
    verbose = cms.untracked.bool(False),
    EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
    ESdigiCollection = cms.InputTag("simEcalPreshowerDigis"),
    moduleLabelMC = cms.string('source')
)


