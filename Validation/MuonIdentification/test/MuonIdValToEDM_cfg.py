import FWCore.ParameterSet.Config as cms

process = cms.Process("MUONIDVALtoEDM")

process.load("DQMServices.Components.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:/tmp/jribnik/bah.root")
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_3XY_V15::All'
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")

process.load("Validation.MuonIdentification.muonIdVal_cff")
process.muonIdVal.baseFolder = cms.untracked.string("Muons/MuonIdVal")
process.muonIdVal.make2DPlots = cms.untracked.bool(False)
process.muonIdDQMInVal.baseFolder = process.muonIdVal.baseFolder

process.load("DQMServices.Components.MEtoEDMConverter_cff")

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("drop *", 
        "keep *_MEtoEDMConverter_*_*", 
        "keep *_*_*_MUONIDVALtoEDM"),
    fileName = cms.untracked.string("file:/tmp/jribnik/meh.root")
)

process.p = cms.Path(process.muonIdValDQMSeq*process.MEtoEDMConverter)
process.e = cms.EndPath(process.out)
