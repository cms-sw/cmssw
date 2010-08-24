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
photonValidation.OutputFileName = 'PhotonValidationRelVal381_SingleGammaPt10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 381 single Photons pt=10GeV
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V8-v1/0011/3ECDAB97-2DA2-DF11-ABC3-001A928116BE.root',
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V8-v1/0010/7657258A-C8A1-DF11-A18B-001A92810AEC.root',
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V8-v1/0010/54E71D91-C9A1-DF11-A992-001A92971B64.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 381 single Photons pt=10GeV
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0011/98D1CE1B-2DA2-DF11-B9EF-0018F3D0967E.root',
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0010/BA03C003-C9A1-DF11-9B15-001A92810A98.root',
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0010/8EAF8E91-C9A1-DF11-88D1-003048D15D04.root',
        '/store/relval/CMSSW_3_8_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V8-v1/0010/0263B70B-C8A1-DF11-B0F1-001A92811716.root' 

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



