import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPValidator")

process.load("FWCore.MessageService.MessageLogger_cfi")
# SIC test
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "START38_V12::All"
#process.options = cms.untracked.PSet(SkipEvent = cms.untracked.vstring('ProductNotFound'))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    #duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring(
       '/store/mc/Fall10/HSCPstop_M-130_7TeV-pythia6/GEN-SIM-RECO/START38_V12-v1/0007/B8FAE5DA-43EE-DF11-95B4-002618943C2D.root',
       '/store/mc/Fall10/HSCPstop_M-130_7TeV-pythia6/GEN-SIM-RECO/START38_V12-v1/0007/2A622861-6FEE-DF11-A544-00E081339049.root',
       '/store/mc/Fall10/HSCPstop_M-130_7TeV-pythia6/GEN-SIM-RECO/START38_V12-v1/0006/D6CE76B9-C5EC-DF11-A422-001A4B0A35FA.root',

    )
)

process.load("PhysicsTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = 'hscpValidatorPlots.root'

process.hscpValidator = cms.EDAnalyzer('HSCPValidator',
  generatorLabel = cms.InputTag("generator"),
  particleIds = cms.vint32(
      # stop R-hadrons
    1000006,
    1000612,
    1000622,
    1000632,
    1000642,
    1000652,
    1006113,
    1006211,
    1006213,
    1006223,
    1006311,
    1006313,
    1006321,
    1006323,
    1006333,
    -1000006,
    -1000612,
    -1000622,
    -1000632,
    -1000642,
    -1000652,
    -1006113,
    -1006211,
    -1006213,
    -1006223,
    -1006311,
    -1006313,
    -1006321,
    -1006323,
    -1006333	  	  
     ),
  EBSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEB"),
  EESimHitCollection = cms.InputTag("g4SimHits","EcalHitsEE"),
  SimTrackCollection = cms.InputTag("g4SimHits"),
  EBDigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
  EEDigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
  RPCRecHitTag = cms.InputTag("rpcRecHits"),
  MakeGenPlots = cms.bool(True),
  MakeSimTrackPlots = cms.bool(True),
  MakeSimDigiPlots = cms.bool(False),
  MakeRecoPlots = cms.bool(True)

)

# SIC test
process.rpcRecHits = cms.EDProducer("RPCRecHitProducer",
    recAlgoConfig = cms.PSet(

    ),
    recAlgo = cms.string('RPCRecHitStandardAlgo'),
#   rpcDigiLabel = cms.InputTag("muonRPCDigis"),
    rpcDigiLabel = cms.InputTag("simMuonRPCDigis"),
    maskSource = cms.string('File'),
    maskvecfile = cms.FileInPath('RecoLocalMuon/RPCRecHit/data/RPCMaskVec.dat'),
    deadSource = cms.string('File'),
    deadvecfile = cms.FileInPath('RecoLocalMuon/RPCRecHit/data/RPCDeadVec.dat')
)

process.p = cms.Path(
# *process.pdigi
#  *process.rpcRecHits
  process.hscpValidator
)
