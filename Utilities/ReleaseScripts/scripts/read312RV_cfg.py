
import FWCore.ParameterSet.Config as cms

process = cms.Process("Rec")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(100) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
                  'rfio:/castor/cern.ch/cms/store/relval/compatibility_ref/CMSSW_3_1_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0007/DAD33187-9078-DE11-9ABE-0019B9F709A4.root'
    )
)

process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )

# output module
#
process.load("Configuration.EventContent.EventContentCosmics_cff")

process.out = cms.OutputModule("PoolOutputModule",
    process.FEVTEventContent,
    fileName = cms.untracked.string('merge.root')
)

# process.source.inputCommands = cms.untracked.vstring(
#          'keep *',
#          'drop *_particleFlowBlock_*_*' )

process.outpath = cms.EndPath(process.out)

