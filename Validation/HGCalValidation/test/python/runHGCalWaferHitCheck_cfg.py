import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
process = cms.Process('HGCGeomAnalysis',Phase2C11)
process.load('Configuration.Geometry.GeometryExtended2026D71_cff')
process.load('Configuration.Geometry.GeometryExtended2026D71Reco_cff')

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Validation.HGCalValidation.hgcWaferHitCheck_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase2_realistic']

if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HGCalValidation')
#   process.MessageLogger.categories.append('HGCalGeom')

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:step1.root')
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.analysis_step = cms.Path(process.hgcalWaferHitCheckEE)
#process.analysis_step = cms.Path(process.hgcalWaferHitCheckHEF)
process.hgcalWaferHitCheckEE.verbosity = 1
process.hgcalWaferHitCheckHEF.verbosity = 1
#process.hgcalWaferHitCheckEE.inputType = 2
#process.hgcalWaferHitCheckHEF.inputType = 2
#process.hgcalWaferHitCheckEE.source = cms.InputTag("simHGCalUnsuppressedDigis", "EE")
#process.hgcalWaferHitCheckHEF.source = cms.InputTag("simHGCalUnsuppressedDigis","HEfront")
#process.hgcalWaferHitCheckEE.inputType = 3                                   
#process.hgcalWaferHitCheckHEF.inputType = 3
#process.hgcalWaferHitCheckEE.source = cms.InputTag("HGCalRecHit", "HGCEERecHits")
#process.hgcalWaferHitCheckHEF.source = cms.InputTag("HGCalRecHit", "HGCHEFRecHits")

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step)
