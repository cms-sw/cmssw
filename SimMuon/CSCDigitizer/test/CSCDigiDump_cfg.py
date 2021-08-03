# TEST CSCDIGIDUMP ON RUN3 SIMULATED DATA - DUMPS THE CSC SIM DIGIS IN A RELVAL FILE
# Based on CSCDigitizerTest_cfg.py once that was working 26.07.2021 - Tim Cox

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('TIM',Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 10 ),
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
##   '/store/relval/CMSSW_12_0_0_pre3/RelValSingleMuPt100/GEN-SIM-DIGI-RAW/120X_mcRun3_2021_realistic_v1-v1/00000/a7d8aebc-928d-419e-973f-0d4ee6dde236.root'
     'file:singlemupt100_10ev.root'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

# Conditions data
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_mc_FULL', '')

# Activate LogVerbatim messages in CSCDigiDump to cout
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout.enable = True
process.MessageLogger.cout.threshold = "INFO"
process.MessageLogger.cout.default = dict( limit = 0 )
process.MessageLogger.cout.INFO = dict (limit = 0 )
process.MessageLogger.cout.CSCDigi = dict( limit = -1 )

## CSCDigiDump - cscDigiDump for real; cscSimDigiDump for sim (both in next cfi)
## Note that MessageLogger category CSCDigi must also be active (see above)
process.load('SimMuon.CSCDigitizer.cscDigiDump_cfi')

## Dump CSC sim digis
process.csc_digi_dump = cms.Path(process.cscSimDigiDump)

process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.csc_digi_dump,process.endjob_step)

