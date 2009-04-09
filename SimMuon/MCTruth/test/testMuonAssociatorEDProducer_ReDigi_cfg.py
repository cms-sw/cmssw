import FWCore.ParameterSet.Config as cms

process = cms.Process("myproc")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:RECOSIM.root'),
    secondaryFileNames = cms.untracked.vstring('file:RAWSIM.root')    
)

# MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")

#process.MessageLogger.debugModules = cms.untracked.vstring("testanalyzer","muonAssociatorByHits","mix")

process.MessageLogger.categories = cms.untracked.vstring('testReader', 'MuonAssociatorEDProducer',
    'MuonAssociatorByHits', 'DTHitAssociator', 'RPCHitAssociator', 'MuonTruth',
    'MixingModule', 'FwkJob', 'FwkReport', 'FwkSummary', 'Root_NoDictionary')

process.MessageLogger.cerr = cms.untracked.PSet(
    noTimeStamps = cms.untracked.bool(True),

    threshold = cms.untracked.string('WARNING'),

    testReader = cms.untracked.PSet(
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
    MixingModule = cms.untracked.PSet(
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

# Mixing Module
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

# Standard Sequences
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load("Configuration.StandardSequences.Digi_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# for 3XX MC data
process.GlobalTag.globaltag = cms.string('IDEAL_30X::All')
# for 2XX MC data
#process.GlobalTag.globaltag = cms.string('IDEAL_V12::All') # 22X
#process.GlobalTag.globaltag = cms.string('IDEAL_V9::All')  # 21X

process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
#process.load('Configuration.EventContent.EventContent_cff')

# MuonAssociatorByHits
process.load("SimMuon.MCTruth.MuonAssociatorByHits_cfi")
process.muonAssociatorByHits.tracksTag = cms.InputTag("globalMuons")
process.muonAssociatorByHits.UseTracker = cms.bool(True)
process.muonAssociatorByHits.UseMuon = cms.bool(True)
#process.muonAssociatorByHits.dumpDT = cms.bool(True)

# test analysis
process.testanalyzer = cms.EDAnalyzer("testReader",
    tracksTag = cms.InputTag("globalMuons"),
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
process.mixing = cms.Path(process.mix)
process.TPs = cms.Path(process.trackingParticles)
process.trackerDigis = cms.Path(process.trDigi)
#process.MuonDigis = cms.Path(process.muonDigi)
process.muonAssociator = cms.Path(process.muonAssociatorByHits)
process.test = cms.Path(process.testanalyzer)
process.output = cms.EndPath(process.MyOut)

#process.digitisation_step = cms.Path(process.pdigi)
process.allDigis = cms.Path(process.trDigi+process.calDigi+process.muonDigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)

# Reinstate TrackingParticles and tracker DigiSimLinks needed for the associator (no ReReconstruction here)
process.schedule = cms.Schedule(process.mixing, process.TPs,
                                process.trackerDigis, #process.MuonDigis,
                                process.muonAssociator, process.test, process.output)

# ReDigi & ReReco
#process.schedule = cms.Schedule(process.mixing, process.TPs,
#                                process.allDigis, process.L1simulation_step, process.digi2raw_step, process.raw2digi_step, process.reconstruction_step, 
#                                process.muonAssociator, process.test, process.output)

def customise(process):
#    process.source.inputCommands = cms.untracked.vstring('keep *','drop FEDRawDataCollection_*_*_*')
#    process.source.dropDescendantsOfDroppedBranches=cms.untracked.bool(False)
      
    if hasattr(process,"RandomNumberGeneratorService"):
        del process.RandomNumberGeneratorService.theSource
    else:    
        process.load("IOMC/RandomEngine/IOMC_cff")
        del process.RandomNumberGeneratorService.theSource

    process.RandomNumberGeneratorService.restoreStateLabel = cms.untracked.string('randomEngineStateProducer')
    
    return(process)

# End of customisation function definition
process = customise(process)
