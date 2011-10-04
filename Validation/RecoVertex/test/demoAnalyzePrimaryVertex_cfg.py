import FWCore.ParameterSet.Config as cms

process = cms.Process("AnalyzePersistentVertices")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring('/store/relval/CMSSW_4_4_0_pre9/RelValQCD_FlatPt_15_3000_N30/GEN-SIM-RECO/DESIGN44_V4_PU_E7TeV_FIX_1_BX156_N30_special_110831-v1/0035/DEEF96F2-BFD3-E011-BEE1-003048679274.root')
)

process.simpleVertexAnalysis = cms.EDAnalyzer("PrimaryVertexAnalyzer",
    simG4 = cms.InputTag("g4SimHits"),
    outputFile = cms.untracked.string('simpleVertexAnalyzer.root'),
    verbose = cms.untracked.bool(False),
    vtxSample = cms.untracked.vstring('offlinePrimaryVertices','offlinePrimaryVerticesWithBS'),
    recoTrackProducer = cms.untracked.string('generalTracks')
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(process.simpleVertexAnalysis*process.dump)


