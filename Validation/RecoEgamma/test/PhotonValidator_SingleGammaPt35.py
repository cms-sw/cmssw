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
process.GlobalTag.globaltag = 'MC_38Y_V8::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal381_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 381single Photons pt=35GeV
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V8-v1/0011/2CB39723-32A2-DF11-BF13-002618943867.root',
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V8-v1/0010/F26673EA-C5A1-DF11-A6B1-002618943831.root',
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V8-v1/0010/7090760E-C5A1-DF11-B86E-002618943875.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 381 single Photons pt=35GeV
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0011/C830492B-32A2-DF11-9ABF-0026189437F8.root',
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0010/E2EB130D-C5A1-DF11-BA23-001A928116C8.root',
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0010/60C9B3DE-C5A1-DF11-86B5-00304867920C.root',
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0010/3A394E0A-C5A1-DF11-BC7C-002618943867.root'
        
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
