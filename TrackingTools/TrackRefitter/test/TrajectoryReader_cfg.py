import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")


process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("TrackingTools.TrackRefitter.globalMuonTrajectories_cff")
process.load("TrackingTools.TrackRefitter.standAloneMuonTrajectories_cff")
process.load("TrackingTools.TrackRefitter.ctfWithMaterialTrajectories_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_V6::All'


process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_4/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0004/10A6F0EE-286C-DD11-AAE9-001617C3B70E.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        TrackTransformer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        TracksToTrajectories = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        TrajectoryReader = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('DEBUG')
    ),
    categories = cms.untracked.vstring('TrackTransformer', 
        'TracksToTrajectories', 
        'TrajectoryReader'),
    destinations = cms.untracked.vstring('cout')
)

process.GLBTrajectoriesReader = cms.EDFilter("TrajectoryReader",
    rootFileName = cms.untracked.string('GLBTajectoriesReader.root'),
    InputLabel = cms.InputTag("globalMuons")
)

process.STATrajectoriesReader = cms.EDFilter("TrajectoryReader",
    rootFileName = cms.untracked.string('STATajectoriesReader.root'),
    InputLabel = cms.InputTag("standAloneMuons")
)

process.CTFTrajectoriesReader = cms.EDFilter("TrajectoryReader",
    rootFileName = cms.untracked.string('CTFTajectoriesReader.root'),
    InputLabel = cms.InputTag("generalTracks")
)

process.glbMuons = cms.Sequence(process.globalMuons*process.GLBTrajectoriesReader)
process.staMuons = cms.Sequence(process.standAloneMuons*process.STATrajectoriesReader)
process.ctf = cms.Sequence(process.ctfWithMaterialTracks*process.CTFTrajectoriesReader)

process.testSTA = cms.Path(process.glbMuons)


