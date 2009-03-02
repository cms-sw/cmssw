import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")

process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#       '/store/relval/CMSSW_3_0_0_pre9/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0002/22CB7289-9AFB-DD11-867C-000423D95030.root'
       '/store/relval/CMSSW_3_0_0_pre9/RelValWjet_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0002/B48A92F2-D3FB-DD11-87B6-001D09F23C73.root'
) )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.simpleVertexAnalysis = cms.EDFilter("KineExample",
    KVFParameters = cms.PSet(
        maxDistance = cms.double(0.01),
        maxNbrOfIterations = cms.int32(10)
    ),
    outputFile = cms.untracked.string('simpleVertexAnalyzer.root'),
    TrackLabel = cms.string('generalTracks')
)

process.p = cms.Path(process.simpleVertexAnalysis)


