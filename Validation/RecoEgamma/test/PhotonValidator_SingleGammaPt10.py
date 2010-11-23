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

photonValidation.OutputFileName = 'PhotonValidationRelVal392_SingleGammaPt10.root'

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
                '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt10/GEN-SIM-RECO/MC_39Y_V3-v1/0073/5EF9B033-A8E9-DF11-B775-0026189437F0.root',
                '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt10/GEN-SIM-RECO/MC_39Y_V3-v1/0067/465CBFE7-14E8-DF11-A6AB-002618943949.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0073/7C668722-A8E9-DF11-88C9-0030486791AA.root',
        '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0067/E80BA26D-14E8-DF11-9C07-001A92971BC8.root',
        '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0067/C0BA55E7-14E8-DF11-B676-001A92971BC8.root',
        '/store/relval/CMSSW_3_9_2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0067/109E34F7-15E8-DF11-B8B3-002618943960.root'


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



