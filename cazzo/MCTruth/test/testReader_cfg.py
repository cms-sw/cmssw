import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:test.root')
)

# MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.categories = cms.untracked.vstring('testReader',
    'FwkJob', 'FwkReport', 'FwkSummary', 'Root_NoDictionary')                                       

process.MessageLogger.cout = cms.untracked.PSet(
    noTimeStamps = cms.untracked.bool(True),
    threshold = cms.untracked.string('INFO'),
    INFO = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    testReader = cms.untracked.PSet(
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

process.MessageLogger.statistics = cms.untracked.vstring('cout')

process.MessageLogger.cerr = cms.untracked.PSet(
    placeholder = cms.untracked.bool(True)
)

process.testanalyzer = cms.EDAnalyzer("testReader",
    tracksTag = cms.InputTag("standAloneMuons"),
    tpTag = cms.InputTag("mix","MergedTrackTruth"),
    assoMapsTag = cms.InputTag("muonAssociatorByHits")
)

process.p = cms.Path(process.testanalyzer)
