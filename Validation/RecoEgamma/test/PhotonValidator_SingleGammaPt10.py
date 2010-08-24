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
process.GlobalTag.globaltag = 'MC_38Y_V9::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre2_SingleGammaPt10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre2 single Photons pt=10GeV
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V9-v1/0015/9E1D7509-6EA8-DF11-8294-0026189438D3.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V9-v1/0013/E8A91176-E4A7-DF11-9304-003048678BEA.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V9-v1/0013/CEDB038F-E4A7-DF11-8D5C-0018F3D09690.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
        # official RelVal 390pre2 single Photons pt=10GeV
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0015/6A645F07-73A8-DF11-8C65-002618943944.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0013/A2D55475-E4A7-DF11-90C8-002618943924.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0013/70C8C273-E4A7-DF11-8539-003048678BEA.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0013/20E56374-E4A7-DF11-8EF0-00261894393E.root'

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



