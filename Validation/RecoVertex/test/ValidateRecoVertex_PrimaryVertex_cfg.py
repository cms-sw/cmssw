import FWCore.ParameterSet.Config as cms

process = cms.Process("AnalyzePersistentVertices")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# Offline primary vertex finding from both track types
#include "RecoVertex/Configuration/data/RecoVertex.cff"
process.load("Validation.RecoVertex.validationPrimaryVertex_cff")



process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0005/E4B3A7BE-3AD7-DE11-9230-002618943939.root')
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


