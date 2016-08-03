import FWCore.ParameterSet.Config as cms

process = cms.Process("HGCGeomAnalysis")

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')    
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2023simReco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
##Global Tag used for production in
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        #'file:testHGCalSimWatcher.root',
        '/store/relval/CMSSW_8_1_0_pre8/RelValTTbar_14TeV/GEN-SIM-RECO/81X_mcRun2_asymptotic_v1_2023LReco-v1/10000/08E719D4-DE3D-E611-9092-003048FFD722.root',
        #'/store/relval/CMSSW_8_1_0_pre6/RelValSingleMuPt100/GEN-SIM-RECO/80X_mcRun2_asymptotic_v14_2023LReco-v1/00000/644ED025-4728-E611-A4BC-0025905A6068.root',
        #'/store/relval/CMSSW_8_1_0_pre4/RelValSingleMuPt100/GEN-SIM-RECO/80X_mcRun2_asymptotic_v13_2023tilted-v1/00000/1C09B4A5-0214-E611-98C8-0CC47A78A3F8.root'
        )
                            )

process.load('Validation.HGCalValidation.test.hgcHitValidation_cfi')

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('RelValTTbarNoPU.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )

#process.hgcHitAnalysis.ietaExcludeBH = [16,92,93,94,95,96,97,98,99,100]
process.hgcHitAnalysis.ietaExcludeBH = [16, 32, 33]

process.p = cms.Path(process.hgcHitAnalysis)


