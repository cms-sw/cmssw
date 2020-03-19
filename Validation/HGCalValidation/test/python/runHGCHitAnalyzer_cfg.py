import FWCore.ParameterSet.Config as cms

#from Configuration.Eras.Era_Phase2C4_cff import Phase2C4
#process = cms.Process('HGCGeomAnalysis',Phase2C4)
#process.load('Configuration.Geometry.GeometryExtended2026D35_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D35Reco_cff')

from Configuration.Eras.Era_Phase2C8_cff import Phase2C8
process = cms.Process('HGCGeomAnalysis',Phase2C8)
process.load('Configuration.Geometry.GeometryExtended2026D41_cff')
process.load('Configuration.Geometry.GeometryExtended2026D41Reco_cff')

#from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
#process = cms.Process('HGCGeomAnalysis',Phase2C9)
#process.load('Configuration.Geometry.GeometryExtended2026D46_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D46Reco_cff')

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')    
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
##Global Tag used for production in
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

#process.MessageLogger.cerr.FwkReport.reportEvery = 100
#if 'MessageLogger' in process.__dict__:
#    process.MessageLogger.categories.append('HGCalValid')

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:step3.root',
        #'root://cms-xrd-global.cern.ch//store/relval/CMSSW_9_0_0_pre1/RelValZEE_14/GEN-SIM-RECO/90X_upgrade2023_realistic_v0_2023D4-v1/10000/085AD7B1-8ABA-E611-A7DA-0CC47A4C8EB6.root',
        #'/store/relval/CMSSW_8_1_0_pre8/RelValTTbar_14TeV/GEN-SIM-RECO/81X_mcRun2_asymptotic_v1_2023LReco-v1/10000/08E719D4-DE3D-E611-9092-003048FFD722.root',
        )
                            )

process.load('Validation.HGCalValidation.hgcHitValidation_cfi')

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('relValTTbar.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )

#process.hgcHitAnalysis.ietaExcludeBH = [16,92,93,94,95,96,97,98,99,100]
#process.hgcHitAnalysis.ietaExcludeBH = [16, 32, 33]

process.p = cms.Path(process.hgcHitAnalysis)


