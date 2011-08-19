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
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
    #duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring(
        '/store/mc/Fall10/HSCPgluino_M-600_7TeV-pythia6/GEN-SIM-RECO/START38_V12-v1/0007/D6965D94-5DEE-DF11-8AF1-0030489454A2.root',
        '/store/mc/Fall10/HSCPgluino_M-600_7TeV-pythia6/GEN-SIM-RECO/START38_V12-v1/0007/C8C0499B-5DEE-DF11-998E-003048C6B52A.root',
        '/store/mc/Fall10/HSCPgluino_M-600_7TeV-pythia6/GEN-SIM-RECO/START38_V12-v1/0007/C81E92A1-92EE-DF11-BDC1-00E081339171.root',
        '/store/mc/Fall10/HSCPgluino_M-600_7TeV-pythia6/GEN-SIM-RECO/START38_V12-v1/0007/7CFA04DA-43EE-DF11-A46F-002618943C2D.root',
        '/store/mc/Fall10/HSCPgluino_M-600_7TeV-pythia6/GEN-SIM-RECO/START38_V12-v1/0007/5EA71CA6-5DEE-DF11-96CD-0030488D0068.root',

#'file:/uscms/home/jchen/rasmusmodel/new/CMSSW_3_9_8/src/gluinoRHadrons.root',
#'file:~/lpcphys/rasmusmodel/PYTHIA6_Exotica_HSCP_gluino600_cfg_GEN_SIM_DIGI_L1_DIGI2RAW_HLT_reggefalse.root'
#'file:/uscmst1b_scratch/lpc1/lpcphys/jchen/rasmusmodel/g600_GENSIMRECO_centralproduction.root'
#'file:/uscms/home/jchen/rasmusmodel/new/CMSSW_3_9_8/src/PYTHIA6_Exotica_HSCP_gluino600_cfg_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root'
#'file:/uscms/home/jchen/rasmusmodel/old/CMSSW_3_9_8/src/PYTHIA6_Exotica_HSCP_gluino600_cfg_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root'
#'file:/uscms/home/jchen/rasmusmodel/old/CMSSW_3_9_8/src/PYTHIA6_Exotica_HSCP_gluino600_cfg_GEN_SIM_DIGI_L1_DIGI2RAW_HLT_reggefalse.root'
#'file:/uscms/home/jchen/rasmusmodel/new/CMSSW_3_9_8/src/PYTHIA6_Exotica_HSCP_gluino600_regge_cfg_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root'
#'file:/uscms/home/jchen/stopbug/CMSSW_3_8_5/src/PYTHIA6_Exotica_HSCP_stop300_cfg_GEN.root'
#'file:/uscmst1b_scratch/lpc1/lpcphys/jchen/rasmusmodel/PYTHIA6_Exotica_HSCP_gluino600_regge_cfg_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root'
)
)

process.load("PhysicsTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = 'hscpValidatorPlots.root'

process.hscpValidator = cms.EDAnalyzer('HSCPValidator',
  generatorLabel = cms.InputTag("generator"),
  particleIds = cms.vint32(
      # stop R-hadrons
##     1000006,
##     1000612,
##     1000622,
##     1000632,
##     1000642,
##     1000652,
##     1006113,
##     1006211,
##     1006213,
##     1006223,
##     1006311,
##     1006313,
##     1006321,
##     1006323,
##     1006333,
##     -1000006,
##     -1000612,
##     -1000622,
##     -1000632,
##     -1000642,
##     -1000652,
##     -1006113,
##     -1006211,
##     -1006213,
##     -1006223,
##     -1006311,
##     -1006313,
##     -1006321,
##     -1006323,
##     -1006333
# gluino
1000021,
1000993,
1009213,
1009313,
1009323,
1009113,
1009223,
1009333,
1091114,
1092114,
1092214,
1092224,
1093114,
1093214,
1093224,
1093314,
1093324,
1093334,
-1009213,
-1009313,
-1009323,
-1091114,
-1092114,
-1092214,
-1092224,
-1093114,
-1093214,
-1093224,
-1093314,
-1093324,
-1093334	
     ),
  EBSimHitCollection = cms.InputTag("g4SimHits","EcalHitsEB"),
  EESimHitCollection = cms.InputTag("g4SimHits","EcalHitsEE"),
  SimTrackCollection = cms.InputTag("g4SimHits"),
  EBDigiCollection = cms.InputTag("simEcalDigis","ebDigis"),
  EEDigiCollection = cms.InputTag("simEcalDigis","eeDigis"),
  RPCRecHitTag = cms.InputTag("rpcRecHits"),
  MakeGenPlots = cms.bool(True),
  MakeSimTrackPlots = cms.bool(False),
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
