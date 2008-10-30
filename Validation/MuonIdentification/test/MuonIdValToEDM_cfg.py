import FWCore.ParameterSet.Config as cms

process = cms.Process("MUONIDVALtoEDM")

process.load("DQMServices.Components.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:/tmp/jribnik/bah.root")
)

process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Validation.MuonIdentification.muonIdVal_cff")
process.muonIdVal.makeEnergyPlots = cms.untracked.bool(True)
process.muonIdVal.makeIsoPlots = cms.untracked.bool(True)
process.muonIdVal.make2DPlots = cms.untracked.bool(True)

process.load("DQMServices.Components.MEtoEDMConverter_cff")

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("drop *", 
        "keep *_MEtoEDMConverter_*_*", 
        "keep *_*_*_MUONIDVALtoEDM"),
    fileName = cms.untracked.string("file:/tmp/jribnik/meh.root")
)

process.p = cms.Path(process.muonIdVal*process.MEtoEDMConverter)
process.e = cms.EndPath(process.out)
