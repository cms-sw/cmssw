import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.GlobalTag.globaltag= "START3X_V25B::All"
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.EventContent.EventContent_cff")

extendAOD = cms.untracked.vstring(
      "drop *",
      "keep *_source_*_*",
      "keep SimTracks_g4SimHits_*_*",
      "keep SimVertexs_g4SimHits_*_*",
      "keep *_offlinePrimaryVertices_*_*",
      "keep *_offlinePrimaryVerticesWithBS_*_*",
      "keep *_offlinePrimaryVerticesDA_*_*",
      "keep recoTracks_generalTracks_*_*")

process.AODSIMEventContent.outputCommands.extend(extendAOD)

process.out = cms.OutputModule("PoolOutputModule",
         outputCommands = process.AODSIMEventContent.outputCommands,
         fileName = cms.untracked.string('pv_reco.root')
)


process.load("RecoVertex.Configuration.RecoVertex_cff")

# the following section is only needed if one wants to modify parameters or the vertexreco sequence
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesWithBS_cfi import *
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesDA_cfi import *
process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesDA_cfi")  # not in the standard configuration
process.vertexreco = cms.Sequence(offlinePrimaryVertices*offlinePrimaryVerticesWithBS*offlinePrimaryVerticesDA)



# primary vertex analyzer(s)
process.load("Validation.RecoVertex.PrimaryVertexAnalyzer_cfi")     # simpleVertexAnalysis
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("Validation.RecoVertex.PrimaryVertexAnalyzer4PU_cfi")  # vertexAnalysis

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100))
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V25B_356ReReco-v1/0004/0E72CE54-F43B-DF11-A06F-0026189438BD.root')
)
process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.p = cms.Path(process.vertexreco*process.vertexAnalysis)

process.outpath=cms.EndPath(process.out)

