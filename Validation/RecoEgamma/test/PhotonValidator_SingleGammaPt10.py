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
process.GlobalTag.globaltag = 'MC_3XY_V21::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal350_SingleGammaPt10.root'
photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 350 single Photons pt=10GeV
        '/store/relval/CMSSW_3_5_0/RelValSingleGammaPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0013/E49BFA38-6213-DF11-AF05-0026189438AA.root',
        '/store/relval/CMSSW_3_5_0/RelValSingleGammaPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0012/AA7F6BC4-2F13-DF11-88B7-001A92971B88.root'
        
    ),
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 350 single Photons pt=10GeV    
        '/store/relval/CMSSW_3_5_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0013/7441E1CB-6213-DF11-BB58-002618943860.root',
        '/store/relval/CMSSW_3_5_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/D2E7CE34-3013-DF11-B165-00304867C04E.root',
        '/store/relval/CMSSW_3_5_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/7C4529AC-2E13-DF11-B7EC-001A92810AEE.root',
        '/store/relval/CMSSW_3_5_0/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/24C3D1B4-2F13-DF11-A3EB-001A92971B68.root' 
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



