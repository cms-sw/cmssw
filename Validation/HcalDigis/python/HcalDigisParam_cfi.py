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
    
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify(hcaldigisAnalyzer,
    dataTPs = cms.InputTag(""),
    digiLabel = cms.string("simHcalDigis")
)
