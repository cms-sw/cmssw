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
process.GlobalTag.globaltag = 'MC_39Y_V6::All'


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

photonValidation.OutputFileName = 'PhotonValidationRelVal394_SingleGammaPt10.root'

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
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt10/GEN-SIM-RECO/MC_39Y_V6-v1/0001/EA2DDAC5-36F8-DF11-B964-0030486790A0.root',
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt10/GEN-SIM-RECO/MC_39Y_V6-v1/0000/5AF40DFF-D9F7-DF11-9BF0-00304867BED8.root'
 

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V6-v1/0001/E05F9523-19F8-DF11-96DD-0026189437ED.root',
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V6-v1/0001/BE8D73BF-33F8-DF11-A75A-0018F3D09696.root',
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V6-v1/0000/501AFDF1-F4F7-DF11-AF3B-001A92971BBE.root',
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V6-v1/0000/24213B09-D4F7-DF11-AFFE-00304867C0FC.root'


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



