import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hcalSimHitStudy = DQMEDAnalyzer('HcalSimHitStudy',
    ModuleLabel = cms.untracked.string('g4SimHits'),
    outputFile = cms.untracked.string(''),
    Verbose = cms.untracked.bool(False),
    HitCollection = cms.untracked.string('HcalHits'),
    TestNumber    = cms.bool(False),
    hep17         = cms.bool(False)
)


from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify( hcalSimHitStudy, ModuleLabel = cms.untracked.string('fastSimProducer') )
    
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
(run2_HCAL_2017 & ~fastSim).toModify( hcalSimHitStudy, TestNumber = cms.bool(True) )

from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toModify( hcalSimHitStudy, hep17 = cms.bool(True) )


