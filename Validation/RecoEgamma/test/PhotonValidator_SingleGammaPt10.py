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
process.GlobalTag.globaltag = 'MC_36Y_V7A::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal361_SingleGammaPt10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 361 single Photons pt=10GeV
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_36Y_V7A-v1/0020/928E6483-FF5C-DF11-8E7C-0018F3D09684.root',
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_36Y_V7A-v1/0020/4A424D68-525D-DF11-B72B-002618943935.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 361 single Photons pt=10GeV    
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V7A-v1/0020/F2238176-FF5C-DF11-98F4-001A92811734.root',
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V7A-v1/0020/9E0C28F4-FF5C-DF11-B35C-0018F3D096CA.root',
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V7A-v1/0020/505D63E4-FE5C-DF11-8547-0026189438E6.root',
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V7A-v1/0020/40CF3B6A-525D-DF11-BD90-0018F3D0965C.root'

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



