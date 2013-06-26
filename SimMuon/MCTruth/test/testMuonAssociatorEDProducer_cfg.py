import FWCore.ParameterSet.Config as cms

process = cms.Process("myproc")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_6_0_pre6/RelValSingleMuPt1000/GEN-SIM-RECO/MC_36Y_V4-v1/0011/72C9B32C-4F45-DF11-AD4A-0026189438F9.root',
    '/store/relval/CMSSW_3_6_0_pre6/RelValSingleMuPt1000/GEN-SIM-RECO/MC_36Y_V4-v1/0010/2E552464-A144-DF11-AD9F-00248C55CC3C.root',
    '/store/relval/CMSSW_3_6_0_pre6/RelValSingleMuPt1000/GEN-SIM-RECO/MC_36Y_V4-v1/0010/2A6E92E9-A244-DF11-BAF2-001A92810AE4.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_6_0_pre6/RelValSingleMuPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0011/9E50692D-4F45-DF11-83CA-0030486792AC.root',
    '/store/relval/CMSSW_3_6_0_pre6/RelValSingleMuPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0011/502CE389-A544-DF11-97B4-00304867902C.root',
    '/store/relval/CMSSW_3_6_0_pre6/RelValSingleMuPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0010/CA650C70-A144-DF11-8A19-003048D25B68.root',
    '/store/relval/CMSSW_3_6_0_pre6/RelValSingleMuPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0010/BCEF716B-A144-DF11-9F01-0018F3D096E8.root',
    '/store/relval/CMSSW_3_6_0_pre6/RelValSingleMuPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0010/54C1A18F-A244-DF11-AF35-0026189438B0.root'
    )
)

# MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")

#process.MessageLogger.debugModules = cms.untracked.vstring("testanalyzer","muonAssociatorByHits","process.muonTrackProducer")

process.MessageLogger.categories = cms.untracked.vstring('testReader', 'MuonAssociatorEDProducer', 'MuonTrackProducer',
    'MuonAssociatorByHits', 'DTHitAssociator', 'RPCHitAssociator', 'MuonTruth',
    'FwkJob', 'FwkReport', 'FwkSummary', 'Root_NoDictionary')

process.MessageLogger.cerr = cms.untracked.PSet(
    noTimeStamps = cms.untracked.bool(True),

    threshold = cms.untracked.string('WARNING'),

    testReader = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    MuonAssociatorEDProducer = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    MuonTrackProducer = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    MuonAssociatorByHits = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    DTHitAssociator = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    RPCHitAssociator = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    MuonTruth = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    )
)

process.MessageLogger.cout = cms.untracked.PSet(
    noTimeStamps = cms.untracked.bool(True),
    
#    threshold = cms.untracked.string('DEBUG'),
    threshold = cms.untracked.string('INFO'),
    
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    testReader = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    MuonAssociatorEDProducer = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    MuonTrackProducer = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    MuonAssociatorByHits = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    DTHitAssociator = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    RPCHitAssociator = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    MuonTruth = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    ),
    FwkReport = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(1),
        limit = cms.untracked.int32(10000000)
    ),
    FwkSummary = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(1),
        limit = cms.untracked.int32(10000000)
    ),
    FwkJob = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    Root_NoDictionary = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    )
)

#process.MessageLogger.statistics = cms.untracked.vstring('cout')

#process.Tracer = cms.Service("Tracer")

# Standard Sequences
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_36Y_V4::All')

# MuonAssociatorByHits
process.load("SimMuon.MCTruth.MuonAssociatorByHits_cfi")
process.muonAssociatorByHits.tracksTag = cms.InputTag("standAloneMuons")
process.muonAssociatorByHits.UseTracker = cms.bool(False)
process.muonAssociatorByHits.UseMuon = cms.bool(True)

# test analysis
process.testanalyzer = cms.EDAnalyzer("testReader",
    tracksTag = cms.InputTag("standAloneMuons"),
    tpTag = cms.InputTag("mix","MergedTrackTruth"),
    assoMapsTag = cms.InputTag("muonAssociatorByHits")
)

# example output
process.MyOut = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep TrackingParticles_mergedtruth_MergedTrackTruth_*',
        'keep *_muonAssociatorByHits_*_*'),
    fileName = cms.untracked.string('test.root')
)

# paths and schedule
process.muonAssociator = cms.Path(process.muonAssociatorByHits)
process.test = cms.Path(process.testanalyzer)
process.output = cms.EndPath(process.MyOut)

process.schedule = cms.Schedule(process.muonAssociator, process.test, process.output)
