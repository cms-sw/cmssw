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
process.GlobalTag.globaltag = 'MC_38Y_V12::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre4_SingleGammaPt10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre4 single Photons pt=10GeV
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V11-v1/0026/C6E90787-67C3-DF11-AB6D-0026189437F5.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V11-v1/0025/6A54EB61-18C3-DF11-95B0-0018F3D09710.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V11-v1/0025/00334166-17C3-DF11-8704-002618943919.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre4 single Photons pt=10GeV
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V11-v1/0026/06613886-67C3-DF11-BFD4-0026189437F2.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V11-v1/0025/C684C664-17C3-DF11-839B-00261894382D.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V11-v1/0025/1C19215D-16C3-DF11-B4D2-0026189437E8.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V11-v1/0025/06BF8364-17C3-DF11-A60B-002618943969.root'


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



