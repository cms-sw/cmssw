import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackOriginAnalyzerTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi.py");
process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.GeometryIdeal_cff")
process.load("SimTracker.TrackHistory.TrackClassifier_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.trackOriginAnalyzer = cms.EDAnalyzer("QualityCutsAnalyzer",
    process.trackClassifier,
    jetTracksAssociation = cms.untracked.InputTag("ic5JetTracksAssociatorAtVertex"),
    # no selection whatsoever   
    minimumNumberOfHits = cms.untracked.int32(0),
    minimumTransverseMomentum = cms.untracked.double(0.0),
    # track categories
    trackQualityClass = cms.untracked.string('loose'),
    minimumNumberOfPixelHits = cms.untracked.int32(0),
    # jetTraksAssociator for optimization analysis
    primaryVertexProducer = cms.untracked.InputTag("offlinePrimaryVertices"),
    maximumChiSquared = cms.untracked.double(10000.0),
    # output file 
    rootFile = cms.untracked.string('file:test.root')
)

process.p = cms.Path(process.trackOriginAnalyzer)

