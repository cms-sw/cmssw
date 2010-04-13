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
process.GlobalTag.globaltag = 'MC_36Y_V4::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre6_SingleGammaPt10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre6 single Photons pt=10GeV
        '/store/relval/CMSSW_3_6_0_pre6/RelValSingleGammaPt10/GEN-SIM-RECO/MC_36Y_V4-v1/0011/F26A7D9A-B044-DF11-81BD-00261894392B.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValSingleGammaPt10/GEN-SIM-RECO/MC_36Y_V4-v1/0011/D052698B-4E45-DF11-A5A3-0030486791DC.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360pre6 single Photons pt=10GeV    
        '/store/relval/CMSSW_3_6_0_pre6/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0011/BA7ADC90-B044-DF11-99D4-002618943906.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0011/B4FACBF0-AF44-DF11-8904-0018F3D09644.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0011/7CF3D17B-4E45-DF11-8FD0-003048679188.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0011/04E35B1B-B344-DF11-BABC-0030486791AA.root'
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



