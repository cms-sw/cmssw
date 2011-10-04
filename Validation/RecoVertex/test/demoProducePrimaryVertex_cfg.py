import FWCore.ParameterSet.Config as cms

process = cms.Process("DempProduce")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.GlobalTag.globaltag= "START44_V4::All"

process.load("Configuration.EventContent.EventContent_cff")


extendAOD = cms.untracked.vstring(
  'drop *',
	    'keep *_source_*_*',
	    'keep *_VtxSmeared_*_*',
	    'keep SimTracks_g4SimHits_*_*',
	    'keep SimVertexs_g4SimHits_*_*',
	    'keep *_offlinePrimaryVertices__Demo',
            'keep recoTracks_generalTracks_*_*')

process.AODSIMEventContent.outputCommands.extend(extendAOD)

process.out = cms.OutputModule("PoolOutputModule",
         outputCommands = process.AODSIMEventContent.outputCommands,
         fileName = cms.untracked.string('reco.root')
)





process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring('/store/relval/CMSSW_4_4_0_pre9/RelValQCD_FlatPt_15_3000_N30/GEN-SIM-RECO/DESIGN44_V4_PU_E7TeV_FIX_1_BX156_N30_special_110831-v1/0035/DEEF96F2-BFD3-E011-BEE1-003048679274.root')
)

process.load("RecoVertex.Configuration.RecoVertex_cff")

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.p = cms.Path(process.vertexreco)
process.outpath = cms.EndPath(process.out)

