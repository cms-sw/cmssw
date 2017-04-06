import FWCore.ParameterSet.Config as cms


hcaldigisAnalyzer = cms.EDAnalyzer("HcalDigisValidation",
    outputFile	= cms.untracked.string(''),
    digiTag	= cms.InputTag("hcalDigis"),
    QIE10digiTag= cms.InputTag("hcalDigis"),
    QIE11digiTag= cms.InputTag("hcalDigis"),
    mode	= cms.untracked.string('multi'),
    hcalselector= cms.untracked.string('all'),
    mc		= cms.untracked.string('yes'),
    simHits     = cms.untracked.InputTag("g4SimHits","HcalHits"),
    emulTPs     = cms.InputTag("emulDigis"),
    dataTPs     = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    TestNumber    = cms.bool(False)
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    hcaldigisAnalyzer.simHits = cms.untracked.InputTag("famosSimHits","HcalHits")

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify(hcaldigisAnalyzer,
    TestNumber    = cms.bool(True)
)
    
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify(hcaldigisAnalyzer,
    dataTPs = cms.InputTag(""),
    digiTag = cms.InputTag("simHcalDigis"),
    QIE10digiTag = cms.InputTag("simHcalDigis","HFQIE10DigiCollection"),
    QIE11digiTag = cms.InputTag("simHcalDigis","HBHEQIE11DigiCollection"),
)
