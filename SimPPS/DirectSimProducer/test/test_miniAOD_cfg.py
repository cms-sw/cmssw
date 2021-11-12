import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
process = cms.Process('PPS', ctpps_2016)

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# minimum of logs
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.threshold = cms.untracked.string('')
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)

# global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '106X_mcRun2_asymptotic_v17', '')

# raw data source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/mc/RunIISummer20UL16MiniAODAPVv2/GGToMuMu_Pt-25_Elastic_13TeV-lpair/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/280000/3870E880-4A47-7440-B122-C76062D2290F.root',
    ),
    #inputCommands = cms.untracked.vstring(
    #    'drop *',
    #    'keep FEDRawDataCollection_*_*_*'
    #)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# PPS simulation and reconstruction chains with standard settings
process.load('SimPPS.Configuration.ppsDirectSim_cff')
process.load('RecoPPS.Configuration.recoCTPPS_cff')

process.p = cms.Path(
    process.ppsDirectSim
    * process.ctppsProtons
)

# output configuration
from RecoPPS.Configuration.RecoCTPPS_EventContent_cff import RecoCTPPSAOD
process.output = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('file:output.root'),
    outputCommands = RecoCTPPSAOD.outputCommands
)

process.outpath = cms.EndPath(process.output)
