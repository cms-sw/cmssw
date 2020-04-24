import FWCore.ParameterSet.Config as cms

process = cms.Process("TestRPCdigi")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

### 2023 Geometry w/ ME0 ###
############################
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_design', '')
# from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixRPCConditions # RPC Conditions for Phase2 Detector (2023)
# process = fixRPCConditions(process)                                                           # RPC Conditions for Phase2 Detector (2023)
# from SimMuon.GEMDigitizer.customizeGEMDigi import customize_digi_addGEM_muon_only             # Customize for CSC + DT + GEM + RPC
# process = customize_digi_addGEM_muon_only(process)                                            # Digi only Muon Detectors
# process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
# process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')
# process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
############################

### 2023 Geometry w/o ME0 ###
############################
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_design', '')
# from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixRPCConditions # RPC Conditions for Phase2 Detector (2023)
# process = fixRPCConditions(process)                                                           # RPC Conditions for Phase2 Detector (2023)
# from SimMuon.GEMDigitizer.customizeGEMDigi import customize_digi_addGEM_muon_only             # Customize for CSC + DT + GEM + RPC
# process = customize_digi_addGEM_muon_only(process)                                            # Digi only Muon Detectors
# process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
# process.load('Configuration.Geometry.GeometryExtended2023_cff')
# process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
#############################

### 2019 Geometry w/ GEM ###
############################
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2019_design', '')
# from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixRPCConditions # RPC Conditions for Phase2 Detector (2019)
# process = fixRPCConditions(process)                                                           # RPC Conditions for Phase2 Detector (2019)
### Info:
### conditions for RPC in 2023 are not updated for higher noises or lower efficiency
### conditions for RPC in 2019 loaded by the global tag are the same as he 2023 conditions
### by default it will not work because it uses the previous RPC Simulation Model (RPCSimAverageNoiseEffCls)
### while the newer RPC Simulation Model (RPCSimAsymmetricCls) is loaded in CMSSW 
### by loading the fixRPCConditions the previous RPC Simulation Model will be loaded
### in future we ll have to give different conditions payload to the 2019 geometry
### such that the newest RPC Simulation Model (RPCSimAsymmetricCls.h) can be used.
# from SimMuon.GEMDigitizer.customizeGEMDigi import customize_digi_addGEM_muon_only   # Customize for CSC + DT + GEM + RPC
# process = customize_digi_addGEM_muon_only(process)                                  # Digi only Muon Detectors
# process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
# process.load('Configuration.Geometry.GeometryExtended2019_cff')
# process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
############################

### Run 2 Geometry ###
######################
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
#  from SimMuon.RPCDigitizer.customizeRPCDigi import customize_digi_muon_only  # Customize for CSC + DT + RPC
# process = customize_digi_muon_only(process)                                 # Digi only Muon Detectors
# process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
# process.load('Configuration.Geometry.GeometryExtended2015_cff')
# process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
######################

### Run 1 Geometry ###
######################
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_mc', '')
from SimMuon.RPCDigitizer.customizeRPCDigi import customize_digi_muon_only  # Customize for CSC + DT + RPC
process = customize_digi_muon_only(process)                                 # Digi only Muon Detectors
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
######################


### Input file  
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
      '/store/relval/CMSSW_7_1_0_pre7/RelValSingleMuPt10/GEN-SIM/PRE_STA71_V3-v1/00000/EC57E8A1-3ED1-E311-8416-002618943882.root' # Run 1 MC
      # '/store/relval/CMSSW_7_4_0_pre7/RelValSingleMuPt10_UP15/GEN-SIM/MCRUN2_74_V7-v1/00000/6A0EACAD-FEB6-E411-90C7-0025905A48FC.root' # Run 2 MC
      # 'file:/afs/cern.ch/work/a/archie/public/SingleMuPt100_GEN-SIM__CMSSW_75X.root' # Upgrade MC (GEN-SIM in 2023 works for all)
    )
)
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( 
        'file:out_digi.root'
    ),
    outputCommands = cms.untracked.vstring(
        'drop *_*_*_*',
        'keep *_*CSC*_*_*',
        'keep *_*DT*_*_*',
        'keep *_*GEM*_*_*',
        'keep *_*RPC*_*_*',
        'keep *_*_*CSC*_*',
        'keep *_*_*DT*_*',
        'keep *_*_*GEM*_*',
        'keep *_*_*RPC*_*',
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('digi_step')
    )
)
process.maxEvents = cms.untracked.PSet( 
    input = cms.untracked.int32(1) 
)


### TO ACTIVATE LogTrace IN RPCDigitizer NEED TO COMPILE IT WITH: 
### --> scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"
### Make sure that you first cleaned your CMSSW version: 
### --> scram b clean 
### before issuing the scram command above
### ------------------------------------------------------------
### LogTrace output goes to cout; all other output to "junk.log"
### Code/Configuration with thanks to Tim Cox
### ------------------------------------------------------------
### to have a handle on the loops inside RPCSimSetup 
### I have split the LogDebug stream in several streams
### that can be activated independently
################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
## process.MessageLogger.categories.append("RPCGeometry")
process.MessageLogger.categories.append("RPCDigiProducer")
## process.MessageLogger.categories.append("RPCSimSetup")
## process.MessageLogger.categories.append("RPCSimSetupClsLoopDetails")
## process.MessageLogger.categories.append("RPCSimSetupNoiseLoopDetails")
## process.MessageLogger.categories.append("RPCSimSetupChecks")
process.MessageLogger.categories.append("RPCSynchronizer")
process.MessageLogger.categories.append("RPCDigitizer")
process.MessageLogger.categories.append("RPCSimAsymmetricCls")
## process.MessageLogger.categories.append("RPCSimAverageNoiseEffCls")
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.destinations = cms.untracked.vstring("cout","junk")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string("DEBUG"),
    default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    ## RPCGeometry = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    RPCDigiProducer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
    ## RPCSimSetup = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    ## RPCSimSetupClsLoopDetails = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    ## RPCSimSetupNoiseLoopDetails = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    ## RPCSimSetupChecks = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    RPCSynchronizer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    RPCDigitizer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    RPCSimAsymmetricCls = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    ## RPCSimAverageNoiseEffCls = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

# process.Timing = cms.Service("Timing")
# process.options = cms.untracked.PSet( 
#     wantSummary = cms.untracked.bool(True) 
# )


process.digi_step     = cms.Path(process.pdigi)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step   = cms.Path(process.endOfProcess)
process.out_step      = cms.EndPath(process.output)


process.schedule = cms.Schedule(
    process.digi_step,
    process.endjob_step,
    process.out_step
)
