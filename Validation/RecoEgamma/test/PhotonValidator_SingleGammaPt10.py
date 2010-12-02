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
process.load("Validation.RecoEgamma.conversionPostprocessing_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_39Y_V5::All'

process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *
from Validation.RecoEgamma.conversionPostprocessing_cfi import *
photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre5_SingleGammaPt10.root'

photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName
photonPostprocessing.OuputFileName = photonValidation.OutputFileName

conversionPostprocessing.standalone = cms.bool(True)
conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName
conversionPostprocessing.OuputFileName = tkConversionValidation.OutputFileName


process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleGammaPt10/GEN-SIM-RECO/MC_39Y_V5-v1/0086/9A6698AB-66F5-DF11-84D3-0018F3D0961A.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleGammaPt10/GEN-SIM-RECO/MC_39Y_V5-v1/0085/76961F72-03F5-DF11-B269-001A92971B3C.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0086/1E94BA0F-70F5-DF11-A622-001A928116FE.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/B85B3565-07F5-DF11-9C91-0018F3D09608.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/92BE8D45-0FF5-DF11-85F1-001A928116C6.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/34BF942F-F9F4-DF11-B273-00304867BFF2.root'
    )
 )


photonPostprocessing.rBin = 48

## For single gamma pt =10
photonValidation.eMax  = 100
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonPostprocessing.eMax  = 100
photonPostprocessing.etMax = 50



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)



process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)



