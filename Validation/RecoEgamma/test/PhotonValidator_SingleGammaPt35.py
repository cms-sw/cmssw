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
process.GlobalTag.globaltag = 'MC_310_V1::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre7_SingleGammaPt35.root'

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
        '/store/relval/CMSSW_3_10_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/MC_310_V1-v1/0103/505D3954-45FD-DF11-96EF-002618FDA21D.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/MC_310_V1-v1/0100/04923A79-CDFC-DF11-89FB-003048678BF4.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_310_V1-v1/0101/405A3329-39FD-DF11-A691-001A92810AD8.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_310_V1-v1/0100/D4F8B779-CDFC-DF11-82A8-003048678BAA.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_310_V1-v1/0100/9E00A87B-CDFC-DF11-AF4E-001A92811730.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_310_V1-v1/0100/002510AF-CEFC-DF11-BFDA-0030486790FE.root'
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



process.p1 = cms.Path(process.tpSelection*process.photonPrevalidationSequence*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
