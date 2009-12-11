import FWCore.ParameterSet.Config as cms

process = cms.Process("AnalyzePersistentVertices")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0005/E4B3A7BE-3AD7-DE11-9230-002618943939.root')
#                            fileNames = cms.untracked.vstring(
#       '/store/relval/CMSSW_3_1_2/RelValTTbar_Tauola_2M/GEN-SIM-RECO/MC_31X_V3-v1/0007/0AB2E447-B478-DE11-8BD8-000423D992DC.root',
#       '/store/relval/CMSSW_3_1_2/RelValTTbar_Tauola_2M/GEN-SIM-RECO/MC_31X_V3-v1/0007/402647C3-B378-DE11-A125-001D09F2447F.root',
#       '/store/relval/CMSSW_3_1_2/RelValTTbar_Tauola_2M/GEN-SIM-RECO/MC_31X_V3-v1/0007/AE71556E-B478-DE11-B8C2-001D09F24637.root'
#    )
)

process.simpleVertexAnalysis = cms.EDAnalyzer("PrimaryVertexAnalyzer",
    simG4 = cms.InputTag("g4SimHits"),
    outputFile = cms.untracked.string('simpleVertexAnalyzer.root'),
    verbose = cms.untracked.bool(False),
    vtxSample = cms.untracked.vstring('offlinePrimaryVertices','offlinePrimaryVerticesWithBS'),
    recoTrackProducer = cms.untracked.string('generalTracks')
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

#process.Tracer = cms.Service("Tracer",
#    indention = cms.untracked.string('$$')
#)

process.p = cms.Path(process.simpleVertexAnalysis*process.dump)


