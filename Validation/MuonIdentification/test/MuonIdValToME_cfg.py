import FWCore.ParameterSet.Config as cms

process = cms.Process("MUONIDVALtoME")

process.load("DQMServices.Components.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:/tmp/jribnik/meh.root")
)

process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.dqmSaver.convention = "Offline" # "RelVal"
process.dqmSaver.workflow = "/Muons/MuonIdVal/RelValSingleMuPt10"
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

process.p = cms.Path(process.EDMtoMEConverter*process.dqmSaver)
