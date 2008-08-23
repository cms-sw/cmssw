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
process.dqmSaver.workflow = "/Muons/MuonIdVal/TEST"

process.p = cms.Path(process.EDMtoMEConverter*process.dqmSaver)
