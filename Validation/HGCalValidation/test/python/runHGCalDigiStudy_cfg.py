import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
process = cms.Process('HGCGeomAnalysis',Phase2C11I13M9)
#process.load('Configuration.Geometry.GeometryExtended2026D77_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D83_cff')
process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Validation.HGCalValidation.hgcDigiStudy_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:step2.root',
        )
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hgcDigiD83tt.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.raw2digi_step = cms.Path(process.RawToDigi)
process.analysis_step = cms.Path(process.hgcalDigiStudyEE+
                                 process.hgcalDigiStudyHEF+
                                 process.hgcalDigiStudyHEB)
process.hgcalDigiStudyEE.verbosity = 1
process.hgcalDigiStudyHEF.verbosity = 1
process.hgcalDigiStudyHEB.verbosity = 1

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.analysis_step)
