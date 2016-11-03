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
    
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify( hcalSimHitStudy, TestNumber = cms.bool(True) )
