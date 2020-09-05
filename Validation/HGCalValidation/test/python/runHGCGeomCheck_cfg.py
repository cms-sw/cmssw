import FWCore.ParameterSet.Config as cms

#from Configuration.Eras.Era_Phase2C4_cff import Phase2C4
#process = cms.Process('HGCGeomAnalysis',Phase2C4)
#process.load('Configuration.Geometry.GeometryExtended2026D35_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D35Reco_cff')

#from Configuration.Eras.Era_Phase2C8_cff import Phase2C8
#process = cms.Process('HGCGeomAnalysis',Phase2C8)
#process.load('Configuration.Geometry.GeometryExtended2026D41_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D41Reco_cff')

#from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
#process = cms.Process('HGCGeomAnalysis',Phase2C9)
#process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')

#from Configuration.Eras.Era_Phase2C12_cff import Phase2C12
#process = cms.Process('HGCGeomAnalysis',Phase2C12)
#process.load('Configuration.Geometry.GeometryExtended2026D58_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D58Reco_cff')

#from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
#process = cms.Process('HGCGeomAnalysis',Phase2C11)
#process.load('Configuration.Geometry.GeometryExtended2026D59_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D59Reco_cff')

from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
process = cms.Process('HGCGeomAnalysis',Phase2C11)
process.load('Configuration.Geometry.GeometryExtended2026D62_cff')
process.load('Configuration.Geometry.GeometryExtended2026D62Reco_cff')

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')    
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HGCalValid')
    process.MessageLogger.categories.append('HGCalGeom')

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testHGCalSimWatcherV14.root',
        )
                            )

process.load('Validation.HGCalValidation.hgcGeomCheck_cff')

process.TFileService = cms.Service("TFileService",
                                fileName = cms.string('hgcGeomCheckD62.root'),
				closeFileFast = cms.untracked.bool(True)
				)

SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )

process.p = cms.Path(process.hgcGeomCheck)


