import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

#  module out = PoolOutputModule 
#  {
#    untracked string fileName = "pv_reco.root"
#    untracked vstring outputCommands = {
#      "drop *",
# Sim info
#      "keep *_source_*_*",
#      "keep SimTracks_g4SimHits_*_*",
#      "keep SimVertexs_g4SimHits_*_*",
# Vertices
#      "keep *_offlinePrimaryVertices_*_*",
# Tracks on input
#      "keep recoTracks_generalTracks_*_*",
#    }
#  }
# Offline primary vertex finding from both track types

extendAOD = cms.untracked.vstring(
      "drop *",
      "keep *_source_*_*",
      "keep SimTracks_g4SimHits_*_*",
      "keep SimVertexs_g4SimHits_*_*",
      "keep *_offlinePrimaryVertices_*_*",
      "keep recoTracks_generalTracks_*_*")

process.AODSIMEventContent.outputCommands.extend(extendAOD)

process.out = cms.OutputModule("PoolOutputModule",
         outputCommands = process.AODSIMEventContent.outputCommands,
         fileName = cms.untracked.string('pv_reco.root')
)

process.load("Configuration.EventContent.EventContent_cff")


process.load("RecoVertex.Configuration.RecoVertex_cff")

# primary vertex analyzer
process.load("Validation.RecoVertex.PrimaryVertexAnalyzer_cfi")

process.source = cms.Source("PoolSource",
    maxEvents = cms.untracked.int32(-1),
    fileNames = cms.untracked.vstring('/store/unmerged/RelVal/2006/9/24/RelVal101Higgs-ZZ-4Mu/GEN-SIM-DIGI-RECO/0005/E2761B69-084C-DB11-9FE2-000E0C3F06E3.root')
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.p = cms.Path(process.vertexreco*process.simpleVertexAnalysis)
process.outpath cms.EndPath(process.out)

