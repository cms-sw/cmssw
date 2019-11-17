import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C8_cff import Phase2C8

process = cms.Process("HcalValid",Phase2C8)
process.load("Configuration.Geometry.GeometryExtended2023D41Reco_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load("Validation.HcalRecHits.hcalDumpHits_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff"
)
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase2_realistic']

process.MessageLogger.cerr.FwkReport.reportEvery = 1
if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HcalValidation')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
#                           fileNames = cms.untracked.vstring('file:step1.root')
#                           fileNames = cms.untracked.vstring('file:step2.root')
                            fileNames = cms.untracked.vstring('file:step3.root')
)

process.p1 = cms.Path(process.hcalDumpHits)
#process.p1 = cms.Path(process.RawToDigi+process.hcalDumpHits)


