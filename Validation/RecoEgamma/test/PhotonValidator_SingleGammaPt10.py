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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre1_SingleGammaPt10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre1 single Photons pt=10GeV
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V8-v1/0009/8C0F452D-079B-DF11-8CCE-002618943834.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V8-v1/0008/D2376560-E19A-DF11-9552-002618943953.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V8-v1/0008/567E9FDD-E19A-DF11-B13A-0026189438F8.root' 
  
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
        # official RelVal 390pre1 single Photons pt=10GeV
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0009/7AAC222E-079B-DF11-B64C-00261894386B.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/B0A57E5F-E19A-DF11-8A5F-003048678FFE.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/92F22A61-E19A-DF11-851C-001A92971B38.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0008/70E6A6DB-E19A-DF11-A80D-002354EF3BE6.root'


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



process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)



