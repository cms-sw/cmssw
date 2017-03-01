import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi")

process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_2_1_8/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0002/04983078-9082-DD11-BB8C-0019DB2F3F9B.root')
)

process.simpleAnalysis = cms.EDAnalyzer("KVFTrackUpdate",
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    TrackLabel = cms.InputTag("standAloneMuons")
)

process.p = cms.Path(process.simpleAnalysis)


