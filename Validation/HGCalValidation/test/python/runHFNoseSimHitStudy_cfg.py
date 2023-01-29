import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C18I13M9_cff import Phase2C18I13M9
process = cms.Process('PROD',Phase2C18I13M9)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load('Configuration.Geometry.GeometryExtended2026D94_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Validation.HGCalValidation.hfnoseSimHitStudy_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalValidation=dict()

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:step1D94.root')
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hfnSimHitD94tt.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.analysis_step   = cms.Path(process.hgcalSimHitStudy)

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step)
