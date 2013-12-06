import FWCore.ParameterSet.Config as cms

hcaldigisAnalyzer = cms.EDAnalyzer("HcalDigisValidation",
    outputFile	= cms.untracked.string(''),
    digiLabel	= cms.InputTag("hcalDigis"),  # regular collections
#--- Two Upgrade (doSLHC=True) collections
    digiLabelHBHE = cms.InputTag("simHcalDigis","HBHEUpgradeDigiCollection"), 
    digiLabelHF	= cms.InputTag("simHcalDigis","HFUpgradeDigiCollection"),
    mode	= cms.untracked.string('multi'),
    hcalselector= cms.untracked.string('all'),
    mc		= cms.untracked.string('yes')
)

