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
process.GlobalTag.globaltag = 'MC_39Y_V2::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre7_SingleGammaPt10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt10/GEN-SIM-RECO/MC_39Y_V2-v1/0045/F877F9A3-B5D3-DF11-9187-002618943843.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt10/GEN-SIM-RECO/MC_39Y_V2-v1/0044/7EF6DC4F-59D3-DF11-850D-0026189438E6.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0045/EC49CB2B-B4D3-DF11-B15D-002618943978.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0044/76CFA8DB-58D3-DF11-AD07-00261894387E.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0044/6EA8454F-59D3-DF11-94F1-001A92811744.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0044/449C094E-59D3-DF11-B6EF-002618943894.root'

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



