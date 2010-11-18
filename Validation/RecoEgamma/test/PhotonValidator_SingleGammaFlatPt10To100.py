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
process.GlobalTag.globaltag = 'MC_38Y_V7::All'

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


photonValidation.OutputFileName = 'PhotonValidationRelVal380_SingleGammaFlatPt10To100.root'

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
# official RelVal 380 single Photons pt=10to100 GeV
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-RECO/MC_38Y_V7-v1/0007/E0129BE0-C495-DF11-9BB9-003048678D52.root',
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-RECO/MC_38Y_V7-v1/0007/5ACA3FEB-C595-DF11-AEBB-00304867926C.root',
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-RECO/MC_38Y_V7-v1/0007/2ADCCE62-C495-DF11-9B1B-00261894396A.root',
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-RECO/MC_38Y_V7-v1/0007/08B89B62-C595-DF11-B204-003048678D52.root'        

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380 single Photons pt=10to100GeV
        
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0007/E2526065-C595-DF11-A3C5-003048678B06.root',
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0007/D033B2E0-C495-DF11-8E41-002618943937.root',
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0007/D0218DE0-C495-DF11-9CF7-003048678C06.root',
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0007/BC3B6364-C495-DF11-841B-0018F3D095F8.root',
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0007/5CCBAAEB-C595-DF11-BB53-0018F3D0968C.root',
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0007/5AD17AEA-C595-DF11-9C4A-002618943886.root',
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0007/4E2D2565-C495-DF11-A6AF-001A92810AD2.root',
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0007/44FF7B64-C495-DF11-B9B9-0018F3D096DA.root',
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0007/34A1F2C6-4E96-DF11-9FE0-001A92971B56.root',
        '/store/relval/CMSSW_3_8_0/RelValSingleGammaFlatPt10To100/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V7-v1/0006/8EB149DD-C295-DF11-9D54-0018F3C3E3A6.root'       
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
