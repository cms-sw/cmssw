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
process.GlobalTag.globaltag = 'MC_38Y_V6::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre8_SingleGammaPt10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre8 single Photons pt=10GeV
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V6-v1/0001/DC7099E6-A78B-DF11-80E4-0030486790A6.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V6-v1/0000/706B529C-6E8B-DF11-B451-002354EF3BD0.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt10/GEN-SIM-RECO/MC_38Y_V6-v1/0000/14C48222-708B-DF11-80EC-001A92810AA0.root' 
 
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre8 single Photons pt=10GeV
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V6-v1/0001/26D67EED-A28B-DF11-83F9-002618943969.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V6-v1/0000/64F67F09-6E8B-DF11-9402-001A92811716.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V6-v1/0000/148A4257-6F8B-DF11-BC1F-002618943836.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V6-v1/0000/10E5DAF9-6E8B-DF11-9DAC-001A92811720.root' 


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



