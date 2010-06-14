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
process.GlobalTag.globaltag = 'MC_37Y_V5::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre1_SingleGammaPt10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre1 single Photons pt=10GeV
  
        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_37Y_V5-v1/0001/260663C9-256E-DF11-867E-002618FDA211.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_37Y_V5-v1/0000/181DC6FB-D76D-DF11-A4F0-0026189438EB.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre1 single Photons pt=10GeV

        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V5-v1/0001/8269B5D5-256E-DF11-B343-0026189438D5.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V5-v1/0000/D02D9810-D86D-DF11-80E2-002618943969.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V5-v1/0000/7E69410A-D86D-DF11-BD9A-002618943866.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V5-v1/0000/5CED1ABE-D96D-DF11-BCD0-0018F3D095F2.root'

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



