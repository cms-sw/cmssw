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
process.GlobalTag.globaltag = 'MC_37Y_V0::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre1_SingleGammaPt10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre1 single Photons pt=10GeV
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_37Y_V0-v1/0015/581B1E69-0E4D-DF11-81C7-00261894398B.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_37Y_V0-v1/0014/E64C49EF-BB4C-DF11-B167-001A928116DA.root'     
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre1 single Photons pt=10GeV    
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0015/FE1F856B-0E4D-DF11-B497-002618943961.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0014/F6C78441-BC4C-DF11-9626-003048678CA2.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0014/8E519257-BD4C-DF11-9A8E-0018F3D09686.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0014/1ECA78A6-BB4C-DF11-818C-00304867BFB2.root' 
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



