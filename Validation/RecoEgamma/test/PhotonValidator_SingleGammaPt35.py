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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre1_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre1 single Photons pt=35GeV
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V8-v1/0009/A09F5F41-079B-DF11-B5B9-003048678B38.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V8-v1/0008/C66372F0-E29A-DF11-B088-001A92971B88.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V8-v1/0008/50D884EE-E29A-DF11-8759-001A928116D6.root'


    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre1 single Photons pt=35GeV

        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0009/AC09C042-079B-DF11-8DB3-002618943860.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/FE3FB3D3-E19A-DF11-8A15-002618943951.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/B8346DE7-E29A-DF11-9D28-003048678FFA.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/7E5446E7-E29A-DF11-9439-003048678B8E.root'

        
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
