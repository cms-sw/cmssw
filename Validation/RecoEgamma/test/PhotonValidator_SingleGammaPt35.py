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
photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre5_SingleGammaPt35.root'

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
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V5-v1/0085/F0B2257F-0EF5-DF11-A259-0018F3D096A2.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V5-v1/0085/96A8B56F-F7F4-DF11-BBC8-001A92811718.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0087/A4C73105-72F5-DF11-90C4-001A928116EA.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/AAAB287A-FEF4-DF11-BF8A-00261894394A.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/906A2A20-F8F4-DF11-85ED-00248C0BE012.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V5-v1/0085/1EECD49E-F6F4-DF11-B7CF-003048678D6C.root'
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
