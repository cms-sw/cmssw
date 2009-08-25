import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.GlobalTag.globaltag= "STARTUP31X_V1::All"
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
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import *
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesWithBS_cfi import *
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesDA_cfi import *

process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVerticesDA_cfi")
process.vertexreco = cms.Sequence(offlinePrimaryVertices*offlinePrimaryVerticesWithBS*offlinePrimaryVerticesDA)

# primary vertex analyzer(s)
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("Validation.RecoVertex.PrimaryVertexAnalyzer_cfi")
process.load("Validation.RecoVertex.PrimaryVertexAnalyzer4PU_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10))
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/w/werdmann/data/311PU/ZMMPU-004.root'))


process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.p = cms.Path(process.vertexreco*process.vertexAnalysis)
#process.p = cms.Path(process.vertexreco*process.simpleVertexAnalysis)
process.outpath=cms.EndPath(process.out)

