import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils

from Configuration.Eras.Era_Phase2C10_cff import Phase2C10
process = cms.Process('PROD',Phase2C10)

process.load('Configuration.Geometry.GeometryExtended2026D60_cff')
process.load('Configuration.Geometry.GeometryExtended2026D60Reco_cff')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.MessageLogger.cerr.FwkReport.reportEvery = 2
if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('HGCalValidation')

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:step3.root',
        )
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load('Validation.HGCalValidation.hfnoseRecHitStudy_cfi')

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hfnRecHitD60tt.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )

process.p = cms.Path(process.hfnoseRecHitStudy)

