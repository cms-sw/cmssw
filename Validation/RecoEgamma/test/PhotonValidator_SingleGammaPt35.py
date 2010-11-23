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
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *
from Validation.RecoEgamma.conversionPostprocessing_cfi import *


photonValidation.OutputFileName = 'PhotonValidationRelVal392_SingleGammaPt35.root'

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
       '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V3-v1/0073/C6BA67E0-A7E9-DF11-8C90-0026189438C2.root',
        '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V3-v1/0067/F2B3CB6E-14E8-DF11-B27E-0018F3D095EE.root',
        '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V3-v1/0067/86E1FB0F-13E8-DF11-8880-00261894388F.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0072/EC422261-96E9-DF11-A90C-0026189438DA.root',
        '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0067/CC0C1257-12E8-DF11-B98E-001A92811724.root',
        '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0067/3A8423FB-12E8-DF11-8EB4-001A92971B28.root',
        '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0067/04DC23E2-13E8-DF11-90D2-001BFCDBD1BA.root'

        
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
