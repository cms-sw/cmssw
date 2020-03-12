import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalRecHitsValidation = DQMEDAnalyzer('EcalRecHitsValidation',
    hitsProducer = cms.string('g4SimHits'),
    outputFile = cms.untracked.string(''),
    EEuncalibrechitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEE"),
    verbose = cms.untracked.bool(False),
    EErechitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    ESrechitCollection = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    EBuncalibrechitCollection = cms.InputTag("ecalMultiFitUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    moduleLabelMC = cms.string('generatorSmeared'),
    enableEndcaps = cms.untracked.bool(True)
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(ecalRecHitsValidation, hitsProducer = "fastSimProducer")
