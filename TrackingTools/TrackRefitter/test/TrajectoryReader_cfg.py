import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")


process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")
process.load("TrackingTools.TrackRefitter.globalMuonTrajectories_cff")
process.load("TrackingTools.TrackRefitter.standAloneMuonTrajectories_cff")
process.load("TrackingTools.TrackRefitter.ctfWithMaterialTrajectories_cff")

#process.globalMuons.TrackTransformer.RefitDirection = 'insideOut'
#process.generalTracks.TrackTransformer.RefitDirection = 'insideOut'
#process.standAloneMuons.TrackTransformer.RefitDirection = 'insideOut'

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string( autoCond[ 'phase1_2022_realistic' ] )

#process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi")

process.source = cms.Source("PoolSource",
#                            skipEvents = cms.untracked.uint32(25),
                            fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_10/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/78E29B63-4699-DD11-AB65-000423D98750.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
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
#    INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    TrackFitters = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    ),
    TrajectoryReader = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
        ),
    threshold = cms.untracked.string('DEBUG')
    ),
                                    categories = cms.untracked.vstring(
    'TrackTransformer', 
    'TracksToTrajectories', 
    'TrackFitters',
    'TrajectoryReader'),
    destinations = cms.untracked.vstring('cout')
)

process.GLBTrajectoriesReader = cms.EDProducer("TrajectoryReader",
                                               rootFileName = cms.untracked.string('GLBTajectoriesReader.root'),
                                               InputLabel = cms.InputTag("globalMuons","Refitted")
                                               )

process.STATrajectoriesReader = cms.EDProducer("TrajectoryReader",
                                               rootFileName = cms.untracked.string('STATajectoriesReader.root'),
                                              InputLabel = cms.InputTag("standAloneMuons","Refitted")
                                               )

process.CTFTrajectoriesReader = cms.EDProducer("TrajectoryReader",
                                             rootFileName = cms.untracked.string('CTFTajectoriesReader.root'),
                                             InputLabel = cms.InputTag("generalTracks","Refitted")
                                             )

process.glbMuons = cms.Sequence(process.globalMuons*process.GLBTrajectoriesReader)
process.staMuons = cms.Sequence(process.standAloneMuons*process.STATrajectoriesReader)
process.tk = cms.Sequence(process.generalTracks*process.CTFTrajectoriesReader)

process.testSTA = cms.Path(process.staMuons+process.tk+process.glbMuons)
#process.testSTA = cms.Path(process.staMuons+process.tk)
#process.testSTA = cms.Path(process.tk)
#process.testSTA = cms.Path(process.staMuons)
