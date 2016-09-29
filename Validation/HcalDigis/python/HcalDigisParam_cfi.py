import FWCore.ParameterSet.Config as cms


hcaldigisAnalyzer = cms.EDAnalyzer("HcalDigisValidation",
    outputFile	= cms.untracked.string(''),
    digiLabel	= cms.string("hcalDigis"),
    mode	= cms.untracked.string('multi'),
    hcalselector= cms.untracked.string('all'),
    mc		= cms.untracked.string('yes'),
    simHits     = cms.untracked.InputTag("g4SimHits","HcalHits"),
    emulTPs     = cms.InputTag("emulDigis"),
    dataTPs     = cms.InputTag("simHcalTriggerPrimitiveDigis")
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    hcaldigisAnalyzer.simHits = cms.untracked.InputTag("famosSimHits","HcalHits")
    
