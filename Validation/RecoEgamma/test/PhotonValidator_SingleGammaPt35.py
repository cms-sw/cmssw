import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidationSequence_cff")
process.load("Validation.RecoEgamma.photonPostprocessing_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_38Y_V9::All'

process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *

photonValidation.OutputMEsInRootFile = True
photonValidation.OutputFileName = 'PhotonValidationRelVal383_SingleGammaPt35.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 383single Photons pt=35GeV
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V9-v1/0022/5C61448C-E9BF-DF11-98F7-001A92971B48.root',
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V9-v1/0022/3E0E3EFF-EDBF-DF11-A36F-00261894394A.root',
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V9-v1/0022/38989D9B-E1BF-DF11-8962-00261894390B.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 383 single Photons pt=35GeV
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0022/E0DB01F4-EDBF-DF11-9287-003048678FB2.root',
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0022/A4FCCA8B-E9BF-DF11-A76C-0018F3D096E4.root',
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0022/3AF7A2F6-E8BF-DF11-A8CA-0018F3D095FA.root',
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0022/28623E82-E1BF-DF11-886D-002354EF3BDA.root',
        '/store/relval/CMSSW_3_8_3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0021/629A8D41-8EBF-DF11-931C-0026189437EB.root'
     
        
    )
 )


photonPostprocessing.rBin = 48

## For single gamma pt = 35
photonValidation.eMax  = 300
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonValidation.dCotCutOn = False
photonValidation.dCotCutValue = 0.15

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)



process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
