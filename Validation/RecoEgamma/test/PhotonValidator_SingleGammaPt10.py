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

photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre2_SingleGammaPt10.root'

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
        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt10/GEN-SIM-RECO/MC_39Y_V3-v1/0059/5A723C4B-E2E2-DF11-BA8B-002618943856.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt10/GEN-SIM-RECO/MC_39Y_V3-v1/0057/F645709C-7FE2-DF11-9DE6-0030486790C0.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0061/164E4B69-F5E2-DF11-AE9C-0018F3D09688.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0057/E6C31421-7FE2-DF11-B0B2-003048678C3A.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0057/4A1B9399-81E2-DF11-BF9B-002618943842.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V3-v1/0057/2270F79B-7FE2-DF11-A851-0026189438E4.root'



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



