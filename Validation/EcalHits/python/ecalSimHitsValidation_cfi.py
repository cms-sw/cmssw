import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalSimHitsValidation = DQMEDAnalyzer("EcalSimHitsValidation",
    ESHitsCollection = cms.string('EcalHitsES'),
    verbose = cms.untracked.bool(False),
    moduleLabelMC = cms.string('generatorSmeared'),
    EBHitsCollection = cms.string('EcalHitsEB'),
    EEHitsCollection = cms.string('EcalHitsEE'),
    moduleLabelG4 = cms.string('g4SimHits')
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(ecalSimHitsValidation, moduleLabelG4 = "fastSimProducer")
