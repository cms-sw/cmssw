import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
HcalSimHitsAnalyser = DQMEDAnalyzer('HcalSimHitsValidation',
    outputFile = cms.untracked.string(''),
    hf1 = cms.double(1/0.383),
    hf2 = cms.double(1/0.368)
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(    HcalSimHitsAnalyser, ModuleLabel = cms.untracked.string("fastSimProducer") )

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
(run2_HCAL_2017 & ~fastSim).toModify( HcalSimHitsAnalyser, TestNumber = cms.untracked.bool(True) )

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal 
phase2_hcal.toModify( HcalSimHitsAnalyser, EEHitCollection = cms.untracked.string("") )

# post-LS1 switch for sampling factors
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( HcalSimHitsAnalyser, 
    hf1 = cms.double(1/0.67),
    hf2 = cms.double(1/0.67)
)
