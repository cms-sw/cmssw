import FWCore.ParameterSet.Config as cms

process = cms.Process("AnalyzePersistentVertices")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# Offline primary vertex finding from both track types
#include "RecoVertex/Configuration/data/RecoVertex.cff"
process.load("Validation.RecoVertex.validationPrimaryVertex_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('dcache:/pnfs/cms/WAX/11/store/RelVal/2007/5/11/RelVal-RelVal_Marcelo_140QCD_pt80_120-1178901791/0000/0C8A65E4-F901-DC11-B65E-00304876A15B.root', 
        'dcache:/pnfs/cms/WAX/11/store/RelVal/2007/5/11/RelVal-RelVal_Marcelo_140QCD_pt80_120-1178901791/0000/24D8B515-5000-DC11-93D3-001731AF685F.root', 
        'dcache:/pnfs/cms/WAX/11/store/RelVal/2007/5/11/RelVal-RelVal_Marcelo_140QCD_pt80_120-1178901791/0000/30E79AB0-2800-DC11-A97A-003048769D5B.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.simplePersistentVertexAnalysis = cms.EDAnalyzer("PrimaryVertexAnalyzer",
    simG4 = cms.InputTag("g4SimHits"),
    outputFile = cms.untracked.string('validation.root'),
    verbose = cms.untracked.bool(True),
    vtxSample = cms.untracked.vstring('offlinePrimaryVerticesFromCTFTracksAVF', 
        'offlinePrimaryVerticesFromCTFTracksKVF'),
    recoTrackProducer = cms.untracked.string('generalTracks')
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.p = cms.Path(process.vertexreco*process.simplePersistentVertexAnalysis)


