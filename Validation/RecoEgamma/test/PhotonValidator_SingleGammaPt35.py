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
process.GlobalTag.globaltag = 'MC_39Y_V3::All'

process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *
from Validation.RecoEgamma.conversionPostprocessing_cfi import *


photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre2_SingleGammaPt35TEST.root'

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

        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V3-v1/0060/6CDA45D3-EFE2-DF11-9B79-002618943972.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V3-v1/0057/A0B54C33-7FE2-DF11-94D8-003048678F78.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V3-v1/0057/66CE66C3-82E2-DF11-BD95-002618943834.root'


    ),
                            
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0061/5EC6D834-F5E2-DF11-84B2-001A92971B72.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0057/EAEAD525-7FE2-DF11-8619-003048678F78.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0057/DC3A20BB-80E2-DF11-8DCD-002618943907.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0057/8454A69C-7FE2-DF11-BFA6-0018F3D09688.root'


        
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



process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
