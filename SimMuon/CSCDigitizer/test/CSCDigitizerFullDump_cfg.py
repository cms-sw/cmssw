# TEST CSCDIGIPRODUCER (CSCDIGITIZER) BY RE-DIGITIZING A RUN3 SIMULATED DATA SAMPLE                                                           
# WITH *FULL DUMP* OF CSCDIGITIZER PROCESS (NOT THE DIGIS, BUT THAT COULD BE ACTIVATED)

# Auto generated configuration file JULY 2021 - Tim Cox 
# - modified until it works on a Run 3 relval file with simhits in 12_0_x.
# - try to minimize stuff in this config file

# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: TEMP4 --step=DIGI --conditions=auto:run3_mc_FULL --filein=/store/relval/CMSSW_12_0_0_pre3/RelValSingleMuPt100/GEN-SIM-DIGI-RAW/120X_mcRun3_2021_realistic_v1-v1/00000/a7d8aebc-928d-419e-973f-0d4ee6dde236.root --mc --era Run3 --no_exec

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('TIM',Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 10 ),
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
##    '/store/relval/CMSSW_12_0_0_pre3/RelValSingleMuPt100/GEN-SIM-DIGI-RAW/120X_mcRun3_2021_realistic_v1-v1/00000/a7d8aebc-928d-419e-973f-0d4ee6dde236.root'
     'file:singlemupt100_10ev.root'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

# Conditions data
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_mc_FULL', '')

# Activate LogTrace messages in CSCDigitizer to cout - 
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout.enable = True
process.MessageLogger.cout.threshold = "DEBUG"
process.MessageLogger.debugModules = ["*"]
process.MessageLogger.cout.default = dict( limit = 0 )
process.MessageLogger.cout.INFO = dict (limit = 0 )
process.MessageLogger.cout.CSCDigitizer = dict( limit = -1 )
process.MessageLogger.cout.CSCStripElectronicsSim = dict( limit = -1 )
process.MessageLogger.cout.CSCWireElectronicsSim = dict( limit = -1 )
process.MessageLogger.cout.CSCBaseElectronicsSim = dict( limit = -1 )
process.MessageLogger.cout.CSCWireHitSim = dict( limit = -1 )
process.MessageLogger.cout.CSCDriftSim = dict( limit = -1 )
process.MessageLogger.cout.CSCCrossGap = dict( limit = -1 )
process.MessageLogger.cout.CSCGasCollisions = dict( limit = -1 )
process.MessageLogger.cout.CSCDigi = dict( limit = -1 )

# CSCDigiDump - cscDigiDump for real; cscSimDigiDump for sim (both in next cfi)
# Note that MessageLogger category CSCDigi must also be active (see above)
## process.load('SimMuon.CSCDigitizer.cscDigiDump_cfi')

# Dump CSC sim digis from the latest process, i.e. the NEW digis not the OLD ones in the input file
## process.csc_digi_dump = cms.Path(process.cscSimDigiDump)

# Activate LogVerbatim output from CSCGasCollisions
process.simMuonCSCDigis.dumpGasCollisions = cms.untracked.bool(True)

# Stuff in a Task is unscheduled - need unscheduled so digitizer will actually run
# => explicitly add CSC digitizer to the path even though it's already part of 'pdigi'
# Still need process.pdigi in order to set up the CrossingFrame of PSimHits for CSCDigiProducer 

process.digi_step = cms.Path(process.pdigi)
process.csc_digi = cms.Path(process.simMuonCSCDigis)
process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.digi_step,process.csc_digi,process.endjob_step)
# Replace the above by the following to add dump of the sim digis produced
## process.schedule = cms.Schedule(process.digi_step,process.csc_digi,process.csc_digi_dump,process.endjob_step)

