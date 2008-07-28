import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")

process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")

process.load("TrackingTools.TrackRefitter.globalMuonTrajectories_cff")

process.load("TrackingTools.TrackRefitter.standAloneMuonTrajectories_cff")

process.load("TrackingTools.TrackRefitter.ctfWithMaterialTrajectories_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/RelVal/2007/7/20/RelVal-RelVal152SingleMuMinusPt10-1184918084/0000/0480715D-B636-DC11-A279-000423D6B1CC.root')
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
    InputLabel = cms.InputTag("ctfWithMaterialTracks")
)

process.glbMuons = cms.Sequence(process.globalMuons*process.GLBTrajectoriesReader)
process.staMuons = cms.Sequence(process.standAloneMuons*process.STATrajectoriesReader)
process.ctf = cms.Sequence(process.ctfWithMaterialTracks*process.CTFTrajectoriesReader)
process.testSTA = cms.Path(process.staMuons)


