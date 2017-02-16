import FWCore.ParameterSet.Config as cms

hcalSimHitStudy = cms.EDAnalyzer("HcalSimHitStudy",
    ModuleLabel = cms.untracked.string('g4SimHits'),
    outputFile = cms.untracked.string(''),
    Verbose = cms.untracked.bool(False),
    HitCollection = cms.untracked.string('HcalHits'),
    TestNumber    = cms.bool(False)
)


from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify( hcalSimHitStudy, ModuleLabel = cms.untracked.string('famosSimHits') )
    
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( hcalSimHitStudy, TestNumber = cms.bool(True) )
