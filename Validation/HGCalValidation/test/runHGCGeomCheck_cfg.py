
import FWCore.ParameterSet.Config as cms

process = cms.Process("HGCGeomAnalysis")

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')    
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')

##v7 Geometry, with hex cells
process.load('Configuration.Geometry.GeometryExtended2023D3Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D3_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
##Global Tag used for production in
## /uscms_data/d3/mhaytmyr/CMSSW_8_0_0_pre4/src/RecoLocalCalo/HGCalRecProducers/test/testHGCalRecoLocal_cfg.py
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testHGCalSimWatcher.root',
        )
                            )

process.load('Validation.HGCalValidation.hgcGeomCheck_cfi')

process.TFileService = cms.Service("TFileService",
                                fileName = cms.string('hgcGeomCheck.root'),
				closeFileFast = cms.untracked.bool(True)
				)

SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )

process.p = cms.Path(process.hgcGeomCheck)


