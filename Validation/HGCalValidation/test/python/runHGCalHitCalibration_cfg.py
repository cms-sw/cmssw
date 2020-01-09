import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils

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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'root://cms-xrd-global.cern.ch//store/relval/CMSSW_9_1_0_pre3/RelValTTbar_14TeV/GEN-SIM/91X_upgrade2023_realistic_v1_D13-v2/10000/0E0708E1-582E-E711-8D30-0025905B8604.root',
        #'root://cms-xrd-global.cern.ch//store/relval/CMSSW_9_1_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU25ns_91X_upgrade2023_realistic_v1_D13PU200-v2/10000/04A22787-5E31-E711-A724-0025905A6090.root',
        #'root://cms-xrd-global.cern.ch//store/relval/CMSSW_9_1_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU25ns_91X_upgrade2023_realistic_v1_D13PU200-v2/10000/06E13ACA-5D31-E711-B32D-0025905A48F2.root',
        #'root://cms-xrd-global.cern.ch//store/relval/CMSSW_9_1_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU25ns_91X_upgrade2023_realistic_v1_D13PU200-v2/10000/0A4780A9-6031-E711-9762-0025905A60F8.root',
        #'root://cms-xrd-global.cern.ch//store/relval/CMSSW_9_1_0_pre3/RelValTTbar_14TeV/GEN-SIM-RECO/PU25ns_91X_upgrade2023_realistic_v1_D13PU200-v2/10000/0A888C70-6531-E711-842E-0CC47A7C346E.root'
        )
                            )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.load('Validation.HGCalValidation.hgcalHitCalibration_cfi')

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('RelVal.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",ignoreTotal = cms.untracked.int32(1) )

process.p = cms.Path(process.hgcalHitCalibration)


