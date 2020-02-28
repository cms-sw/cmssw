import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C6_cff import Phase2C6
process = cms.Process('PROD',Phase2C6)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('Configuration.Geometry.GeometryExtended2026D44_cff')
process.load('Configuration.Geometry.GeometryExtended2026D44Reco_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Validation.HGCalValidation.hfnoseDigiStudy_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase2_realistic']

if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HGCalValidation')
    process.MessageLogger.categories.append('HGCalGeom')

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:step2.root',
        )
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hfnDigiD44tt.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.raw2digi_step = cms.Path(process.RawToDigi)
process.analysis_step = cms.Path(process.hfnoseDigiStudy)
process.hfnoseDigiStudy.verbosity = 1

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.analysis_step)
