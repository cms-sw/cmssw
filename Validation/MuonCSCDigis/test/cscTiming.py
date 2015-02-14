import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD2")

#process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.Sim_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("SimMuon.CSCDigitizer.muonCSCDigis_cfi")
# if the data file doesn't have reco
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.DigiToRaw_cff")
process.load("Configuration.StandardSequences.SimL1Emulator_cff")
process.load("L1Trigger.CSCTriggerPrimitives.test.CSCTriggerPrimitivesReader_cfi")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Validation.CSCRecHits.cscRecHitValidation_cfi")
process.load("Validation.MuonCSCDigis.cscDigiValidation_cfi")

process.cscValidation = cms.EDFilter("CSCValidation",
    # name of file which will contain output
    rootFileName = cms.untracked.string('validationHists_sim.root'),
    # basically turns on/off residual plots which use simhits
    isSimulation = cms.untracked.bool(True),
    # stores a tree of info for the first 1.5M rechits and 2M segments
    # used to make 2D scatter plots of global positions.  Significantly increases
    # size of output root file, so beware...
    writeTreeToFile = cms.untracked.bool(True),
    # mostly for MC and RECO files which may have dropped the digis
    useDigis = cms.untracked.bool(True),
    # lots of extra, more detailed plots
    detailedAnalysis = cms.untracked.bool(False),
    # Input tags for various collections CSCValidation looks at
    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    compDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    cscRecHitTag = cms.InputTag("csc2DRecHits"),
    cscSegTag = cms.InputTag("cscSegments"),
    # set to true to only look at events with CSC L1A
    useTrigger = cms.untracked.bool(False),
    # set to true to skip "messy" events
    filterCSCEvents = cms.untracked.bool(False),
    # do you want to look at STA muons?
    makeStandalonePlots = cms.untracked.bool(False),
    # STA tag for cosmics
    saMuonTag = cms.InputTag("cosmicMuonsEndCapsOnly"),
    l1aTag = cms.InputTag("gtDigis"),
    simHitTag = cms.InputTag("g4SimHits", "MuonCSCHits")
)

#process.MessageLogger = cms.Service("MessageLogger",
#    destinations = cms.untracked.vstring("debug"),
    #   untracked vstring categories     = { "lctDigis" }
    #   untracked vstring debugModules   = { "*" }
    #   untracked PSet debugmessages.txt = {
    #       untracked string threshold = "DEBUG"
    #       untracked PSet INFO     = {untracked int32 limit = 0}
    #       untracked PSet DEBUG    = {untracked int32 limit = 0}
    #       untracked PSet lctDigis = {untracked int32 limit = 10000000}
    #   }
#    debug = cms.untracked.PSet(
#        threshold = cms.untracked.string("DEBUG"),
#        extension = cms.untracked.string(".txt"),
#        lineLength = cms.untracked.int32(132),
#        noLineBreaks = cms.untracked.bool(True)
#    ),
#    debugModules = cms.untracked.vstring("lctreader")
#)


process.GlobalTag.globaltag = "MC_3XY_V12::All"
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(250))

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(False),
    debugVebosity = cms.untracked.uint32(20),
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_4_0_pre4/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V12-v1/0000/F842BE1A-7FC8-DE11-A2D8-00304879FBB2.root'
#'/store/relval/CMSSW_3_4_0_pre2/RelValSingleMuPt100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/7295A693-C0BD-DE11-90FD-00248C0BE01E.root'
), duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

process.mu = cms.EDAnalyzer("Mu")
process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.load("SimMuon.CSCDigitizer.cscDigiDump_cfi")
process.cscDigiDump.stripDigiTag = "muonCSCDigis:MuonCSCStripDigi"
process.cscDigiDump.wireDigiTag = "muonCSCDigis:MuonCSCWireDigi"

process.p = cms.Path(process.mix+process.simMuonCSCDigis+ process.SimL1Emulator+process.cscpacker+process.rawDataCollector+process.muonCSCDigis+process.muonDTDigis+process.muonRPCDigis + process.muonlocalreco+process.cscDigiValidation+process.cscRecHitValidation)

