
import FWCore.ParameterSet.Config as cms

process = cms.Process("validation")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryIdeal_cff")
process.load("RecoBTag.Analysis.bTagAnalysis_cfi")

process.load("SimTracker.TrackHistory.JetVetoedTracksAssociator_cfi")

process.btag = cms.Path(process.btagging)
process.plots = cms.Path(process.bTagAnalysis)
process.jassoc = cms.Path(process.ic5JetVetoedTracksAssociatorAtVertex)
process.schedule = cms.Schedule(process.jassoc, process.btag, process.plots)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.bTagAnalysis.producePs = False

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.bTagAnalysis.rootfile = cms.string('test.root')
process.impactParameterTagInfos.jetTracks = cms.InputTag("ic5JetVetoedTracksAssociatorAtVertex")

