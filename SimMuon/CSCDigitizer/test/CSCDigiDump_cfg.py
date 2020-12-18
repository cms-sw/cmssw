# Based on cmsDriver.py created file - but needed a LOT of editing - Tim Cox, Feb/Mar 2015
# 05.03.2015 - runs in 73x-75x (at least) - dumps all CSC digis, but not CSCDigitizer debugging info

import FWCore.ParameterSet.Config as cms

process = cms.Process('DIGI')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')

process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 100 )
)

process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_7_3_0/RelValSingleMuPt100_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/MCRUN2_73_V7-v1/00000/0864952B-5781-E411-99D5-0025905964CC.root'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# MessageLogger

process.load('FWCore.MessageService.MessageLogger_cfi')

# Activate LogVerbatim output in CSCDigitizer
# Activate LogVerbatim output in CSC Digis and CSCDigiDump


process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
##    CSCDigitizer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    CSCDigi      = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

## To dump CSC digis need to include CSCDigiDump, AND activate LogVerbatim for "CSCDigi" above
process.load("SimMuon.CSCDigitizer.cscDigiDump_cfi")
# CSCDigiDump - cscDigiDump for real; cscSimDigiDump for sim (both in above cfi)
##process.digi = cms.Path(process.pdigi)
process.digi = cms.Path(process.pdigi*process.cscSimDigiDump)
process.endjob = cms.EndPath(process.endOfProcess)
process.schedule = cms.Schedule(process.digi,process.endjob)



