import FWCore.ParameterSet.Config as cms

process = cms.Process("myproc")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_3_3/RelValSingleMuPt100/GEN-SIM-RECO/MC_31X_V9-v2/0001/966611DE-FCD1-DE11-A94C-0030486792F0.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_3_3/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0002/969130B8-32D2-DE11-8BA7-002618943986.root',
    '/store/relval/CMSSW_3_3_3/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0001/A690D8C8-F7D1-DE11-97E4-002354EF3BD0.root',
    '/store/relval/CMSSW_3_3_3/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0001/906ABADC-FCD1-DE11-A328-003048679076.root'
    )
)

# MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")

#process.MessageLogger.debugModules = cms.untracked.vstring("testanalyzer","muonAssociatorByHits")

process.MessageLogger.categories = cms.untracked.vstring('testReader', 'MuonAssociatorEDProducer',
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
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V9::All')

# MuonAssociatorByHits
process.load("SimMuon.MCTruth.MuonAssociatorByHits_cfi")
process.muonAssociatorByHits.tracksTag = cms.InputTag("standAloneMuons")
process.muonAssociatorByHits.UseTracker = cms.bool(False)
process.muonAssociatorByHits.UseMuon = cms.bool(True)

# test analysis
process.testanalyzer = cms.EDAnalyzer("testReader",
    tracksTag = cms.InputTag("standAloneMuons"),
    tpTag = cms.InputTag("mergedtruth","MergedTrackTruth"),
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
