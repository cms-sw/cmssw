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
process.GlobalTag.globaltag = "START3X_V26::All"

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(25) )

process.source = cms.Source("PoolSource",
    #duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring(
      'rfio:/castor/cern.ch/user/s/scooper/hscp/354/MGStop130-GEN-SIM_10.root'
    )
)

process.load("PhysicsTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = 'hscpValidatorPlots.root'

process.hscpValidator = cms.EDAnalyzer('HSCPValidator',
  generatorLabel = cms.InputTag("generator"),
  particleIds = cms.vint32(
      # stop R-hadrons
      1000006,
      -1000006
     ),
  EBSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEB"),
  EESimHitCollection = cms.InputTag("g4SimHits","EcalHitsEE"),
  SimTrackCollection = cms.InputTag("g4SimHits"),
  EBDigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
  EEDigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
  RPCRecHitTag = cms.InputTag("rpcRecHits"),
  MakeGenPlots = cms.bool(True),
  MakeSimDigiPlots = cms.bool(True),
  MakeRecoPlots = cms.bool(False)

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
  process.pdigi
  *process.rpcRecHits
  *process.hscpValidator
)
