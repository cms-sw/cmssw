import FWCore.ParameterSet.Config as cms

cscDigiValidation = cms.EDAnalyzer("CSCDigiValidation",
    simHitsTag = cms.InputTag("mix", "g4SimHitsMuonCSCHits"),
    wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    outputFile = cms.string(''),
    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    alctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    clctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    doSim = cms.bool(False)
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    cscDigiValidation.simHitsTag = cms.InputTag("mix", "MuonSimHitsMuonCSCHits")


