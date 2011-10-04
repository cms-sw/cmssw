import FWCore.ParameterSet.Config as cms

process = cms.Process("AnalyzePersistentVertices")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.GlobalTag.globaltag= "START44_V4::All"


# Offline primary vertex finding from both track types
#include "RecoVertex/Configuration/data/RecoVertex.cff"
process.load("Validation.RecoVertex.validationPrimaryVertex_cff")



process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/relval/CMSSW_4_4_0_pre9/RelValQCD_FlatPt_15_3000_N30/GEN-SIM-RECO/DESIGN44_V4_PU_E7TeV_FIX_1_BX156_N30_special_110831-v1/0035/DEEF96F2-BFD3-E011-BEE1-003048679274.root')
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.simplePersistentVertexAnalysis = cms.EDAnalyzer("PrimaryVertexAnalyzer",
    simG4 = cms.InputTag("g4SimHits"),
    outputFile = cms.untracked.string('validation.root'),
    verbose = cms.untracked.bool(True),
    vtxSample = cms.untracked.vstring('offlinePrimaryVertices', 
        'offlinePrimaryVerticesBS'),
    recoTrackProducer = cms.untracked.string('generalTracks')
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.p = cms.Path(process.vertexreco*process.simplePersistentVertexAnalysis)


