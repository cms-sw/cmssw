import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
simHitsValidationHcal = DQMEDAnalyzer('SimHitsValidationHcal',
    ModuleLabel   = cms.string('g4SimHits'),
    HitCollection = cms.string('HcalHits'),
    Verbose       = cms.bool(False),
    TestNumber    = cms.bool(False),
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify( simHitsValidationHcal, ModuleLabel = cms.string("fastSimProducer") )

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
(run2_HCAL_2017 & ~fastSim).toModify( simHitsValidationHcal, TestNumber = cms.bool(True) )
