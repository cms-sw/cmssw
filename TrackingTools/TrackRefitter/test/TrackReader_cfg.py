import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")


process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")

process.load("RecoMuon.Configuration.RecoMuon_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.GeometryDB_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string( autoCond[ 'phase1_2022_realistic' ] )
process.load("Configuration.StandardSequences.RawToDigi_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_9/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/2E29E000-A985-DD11-8C04-000423D98950.root')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
    noLineBreaks = cms.untracked.bool(True),
    DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TrackReader = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('DEBUG')
        ),
                                    categories = cms.untracked.vstring( 
    'TrackReader'),
    destinations = cms.untracked.vstring('cout')
)

process.TrackReader = cms.EDAnalyzer("TrackReader",
                                   InputLabel = cms.InputTag("generalTracks"),
                                   TrackerRecHitBuilder = cms.string('WithAngleAndTemplate'),
                                   MuonRecHitBuilder = cms.string('MuonRecHitBuilder')
                                   )


process.testSTA = cms.Path(process.TrackReader)


