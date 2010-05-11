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
process.GlobalTag.globaltag = 'MC_37Y_V2::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre3_SingleGammaPt10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre3 single Photons pt=10GeV
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt10/GEN-SIM-RECO/MC_37Y_V2-v1/0019/4CC02B1C-F057-DF11-93FF-002354EF3BE0.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt10/GEN-SIM-RECO/MC_37Y_V2-v1/0019/4C4C73BC-3558-DF11-8B34-003048679006.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre3 single Photons pt=10GeV    
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V2-v1/0019/B40D9FC5-F057-DF11-8FFD-001A92971B94.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V2-v1/0019/78E82A2C-F157-DF11-B469-0018F3D095FA.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V2-v1/0019/24DC688D-3458-DF11-9B45-003048678E24.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V2-v1/0018/9A51D83A-E657-DF11-8842-001A92971B0E.root'

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



