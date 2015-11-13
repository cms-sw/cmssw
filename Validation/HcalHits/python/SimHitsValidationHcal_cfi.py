import FWCore.ParameterSet.Config as cms

simHitsValidationHcal = cms.EDAnalyzer("SimHitsValidationHcal",
    ModuleLabel   = cms.untracked.string('g4SimHits'),
    HitCollection = cms.untracked.string('HcalHits'),
    Verbose       = cms.untracked.bool(False),
    TestNumber    = cms.untracked.bool(False),
)

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    simHitsValidationHcal.ModuleLabel = cms.untracked.string("famosSimHits")
    


