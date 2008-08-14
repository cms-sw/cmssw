import FWCore.ParameterSet.Config as cms

process = cms.Process("DempProduce")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.EventContent.EventContent_cff")


extendAOD = cms.untracked.vstring(
  'drop *',
	    'keep *_source_*_*',
	    'keep *_VtxSmeared_*_*',
	    'keep SimTracks_g4SimHits_*_*',
	    'keep SimVertexs_g4SimHits_*_*',
	    'keep *_offlinePrimaryVertices_*_Demo',
            'keep recoTracks_generalTracks_*_*')

process.AODSIMEventContent.outputCommands.extend(extendAOD)

process.out = cms.OutputModule("PoolOutputModule",
         outputCommands = process.AODSIMEventContent.outputCommands,
         fileName = cms.untracked.string('reco.root')
)


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )


process.load("RecoVertex.Configuration.RecoVertex_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:pv_reco.root')
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.p = cms.Path(process.vertexreco)
process.outpath = cms.EndPath(process.out)

