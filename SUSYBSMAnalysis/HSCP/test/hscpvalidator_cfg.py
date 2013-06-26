import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPValidator")

process.load("FWCore.MessageService.MessageLogger_cfi")
# SIC test
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "START311_V2::All"
#process.options = cms.untracked.PSet(SkipEvent = cms.untracked.vstring('ProductNotFound'))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
    #duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring(
#       '/store/user/jchen/HSCPdichampSummer11/BX1/DIGI/DC100/jiechen/EXO_HSCP_Dichamp100_Summer11BX1Reproduce_GEN/EXO_HSCP_Dichamp100_Summer11BX1Reproduce_HLT/99a4e336940ec729f14f9c623a8b05f3/REDIGI_DIGI_L1_DIGI2RAW_HLT_PU_9_1_G6X.root',
#'file:/uscms/home/jchen/temp/CMSSW_4_1_5/src/HSCPhip1_M_200_7TeV_pythia6_cff_py_RAW2DIGI_RECO.root'
'file:g800_reco.root',
)
)

process.load("PhysicsTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = 'hscpValidatorPlots.root'

process.hscpValidator = cms.EDAnalyzer('HSCPValidator',
  generatorLabel = cms.InputTag("generator"),
  particleIds = cms.vint32(
	# stop R-hadrons
	#1000612,    1000622,    1000632,    1000642,    1000652,    1006113,    1006211,    1006213,    1006223,    1006311,    1006313,    1006321,    1006323,    1006333,    -1000006,    -1000612,    -1000622,    -1000632,    -1000642,    -1000652,    -1006113,    -1006211,    -1006213,    -1006223,    -1006311,    -1006313,    -1006321,    -1006323,    -1006333
# gluino
1000993,1009213,1009313,1009323,1009113,1009223,1009333,1091114,1092114,1092214,1092224,1093114,1093214,1093224,1093314,1093324,1093334,-1009213,-1009313,-1009323,-1091114,-1092114,-1092214,-1092224,-1093114,-1093214,-1093224,-1093314,-1093324,-1093334
#Stau
# 	1000015, 	-1000015, 	2000015, -2000015,
#tau'
# 	17, -17
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
  MakeRecoPlots = cms.bool(False),
  MakeHLTPlots = cms.bool(True)

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
# process.pdigi*process.rpcRecHits
#  *process.hscpValidator
	process.hscpValidator
)
