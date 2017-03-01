import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(3) )
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()

process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( (
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v2/0005/2A400E33-B0E2-DD11-94F7-000423D60FF6.root',
) );

secFiles.extend( (
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/22F84CC1-B0E2-DD11-97FA-001617E30D00.root',
       '/store/relval/CMSSW_3_0_0_pre6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v2/0005/447729C6-B0E2-DD11-A066-001617C3B6C6.root',
) )


# Track Associators
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load("SimTracker.VertexAssociation.VertexAssociatorByTracks_cfi")
process.load("RecoTracker.Configuration.RecoTracker_cff")

process.demo = cms.EDAnalyzer('testVertexAssociator')
process.p = cms.Path(process.demo)
