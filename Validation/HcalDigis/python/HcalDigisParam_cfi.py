import FWCore.ParameterSet.Config as cms


hcaldigisAnalyzer = cms.EDAnalyzer("HcalDigisValidation",
    outputFile	= cms.untracked.string(''),
    digiLabel	= cms.InputTag("hcalDigis"),
    mode	= cms.untracked.string('multi'),
    hcalselector= cms.untracked.string('all'),
    mc		= cms.untracked.string('yes'),
    simHits     = cms.untracked.InputTag("g4SimHits","HcalHits")
)

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    hcaldigisAnalyzer.simHits = cms.untracked.InputTag("famosSimHits","HcalHits")
    
