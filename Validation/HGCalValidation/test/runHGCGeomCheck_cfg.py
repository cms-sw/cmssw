import FWCore.ParameterSet.Config as cms

#from Configuration.Eras.Era_Phase2_cff import Phase2
#process = cms.Process("HGCGeomAnalysis",Phase2)
#process.load('Configuration.Geometry.GeometryExtended2023D21Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D21_cff')
#from Configuration.Eras.Era_Phase2C4_cff import Phase2C4
#process = cms.Process("HGCGeomAnalysis",Phase2C4)
#process.load('Configuration.Geometry.GeometryExtended2023D28Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D28_cff')
from Configuration.Eras.Era_Phase2C4_timing_layer_bar_cff import Phase2C4_timing_layer_bar
process = cms.Process("HGCGeomAnalysis",Phase2C4_timing_layer_bar)
process.load('Configuration.Geometry.GeometryExtended2023D41Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D41_cff')

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
        'file:testHGCalSimWatcherV10.root',
        )
                            )

process.load('Validation.HGCalValidation.hgcGeomCheck_cff')

process.TFileService = cms.Service("TFileService",
                                fileName = cms.string('hgcGeomCheckV10.root'),
				closeFileFast = cms.untracked.bool(True)
				)

SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )

process.p = cms.Path(process.hgcGeomCheck)


