import FWCore.ParameterSet.Config as cms

hcalSimHitStudy = cms.EDAnalyzer("HcalSimHitStudy",
    ModuleLabel = cms.untracked.string('g4SimHits'),
    outputFile = cms.untracked.string(''),
    Verbose = cms.untracked.bool(False),
    HitCollection = cms.untracked.string('HcalHits')
)


from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify( hcalSimHitStudy, ModuleLabel = cms.untracked.string('famosSimHits') )
    
