import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
HcalSimHitsAnalyser = DQMEDAnalyzer('HcalSimHitsValidation',
    outputFile = cms.untracked.string('')
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(    HcalSimHitsAnalyser, ModuleLabel = cms.untracked.string("famosSimHits") )

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( HcalSimHitsAnalyser, TestNumber = cms.untracked.bool(True), EEHitCollection = cms.untracked.string("") )
