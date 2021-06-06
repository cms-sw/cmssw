import FWCore.ParameterSet.Config as cms

process = cms.Process("myproc")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
#
# input dataset is a SuperPointing Skim
    fileNames = cms.untracked.vstring('file:SuperPointingSkim_GEN-SIM-RAW-RECO.root')                     
#
# input dataset is two-files GEN-SIM-RECO and GEN-SIM-RAW
#    fileNames = cms.untracked.vstring('file:RECOSIM.root'),
#    secondaryFileNames = cms.untracked.vstring('file:RAWSIM.root')    
#
)

# MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")

#process.MessageLogger.debugModules = cms.untracked.vstring("testanalyzer","muonAssociatorByHits","process.muonTrackProducer")
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
    enable = cms.untracked.bool(True),
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
    MixingModule = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
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

#process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)

#process.Tracer = cms.Service("Tracer")

# Standard Sequences
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.DigiCosmics_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# set the GlobalTag according to the input MC sample
#
process.GlobalTag.globaltag = cms.string('COSMC_42_PEAB::All')
#process.GlobalTag.globaltag = cms.string('COSMC_42_DECB::All')

process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.ReconstructionCosmics_cff')
process.load('Configuration.EventContent.EventContentCosmics_cff')

# MuonAssociatorByHits
process.load("SimMuon.MCTruth.MuonAssociatorByHits_cfi")
#process.muonAssociatorByHits.tracksTag = cms.InputTag("cosmicMuons1Leg")
#process.muonAssociatorByHits.UseTracker = cms.bool(False)
#process.muonAssociatorByHits.UseMuon = cms.bool(True)
process.muonAssociatorByHits.tracksTag = cms.InputTag("globalCosmicMuons1Leg")
process.muonAssociatorByHits.UseTracker = cms.bool(True)
process.muonAssociatorByHits.UseMuon = cms.bool(True)

# test analysis
process.testanalyzer = cms.EDAnalyzer("testReader",
    tpTag = process.muonAssociatorByHits.tpTag,
    tracksTag = process.muonAssociatorByHits.tracksTag,
    assoMapsTag = cms.InputTag("muonAssociatorByHits")
)

# example output
process.MyOut = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep recoTracks_cosmicMuons1Leg_*_*', 
        'keep recoTracks_globalCosmicMuons1Leg_*_*', 
        'keep TrackingParticles_mergedtruth_MergedTrackTruth_*',
        'keep *_muonAssociatorByHits_*_*'),
    fileName = cms.untracked.string('test.root')
)

# restore random number generator seeds corresponding to the input events
process.RandomNumberGeneratorService.restoreStateLabel = cms.untracked.string('randomEngineStateProducer')

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
process.reconstruction_step = cms.Path(process.reconstructionCosmics)


# Reinstate TrackingParticles and tracker DigiSimLinks needed for the associator (no ReReconstruction here)
process.schedule = cms.Schedule(process.mixing, process.TPs,
                                process.trackerDigis, #process.MuonDigis,
                                process.muonAssociator, process.test, process.output)

# ReDigi & ReReco
#process.schedule = cms.Schedule(process.mixing, process.TPs,
#                                process.allDigis, process.L1simulation_step, process.digi2raw_step, process.raw2digi_step, process.reconstruction_step, 
#                                process.muonAssociator, process.test, process.output)
