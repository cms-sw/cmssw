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

photonValidation.OutputFileName = 'PhotonValidationRelVal352_SingleGammaPt10.root'
photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

# official RelVal 352 single Photons pt=10GeV
        '/store/relval/CMSSW_3_5_2/RelValSingleGammaPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0016/48DE9A6A-D91E-DF11-9436-003048678B0C.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleGammaPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0015/3431C3CA-1C1E-DF11-8A94-001731AF6B79.root'      
    ),
                            
    secondaryFileNames = cms.untracked.vstring(


# official RelVal 352 single Photons pt=10GeV    
        '/store/relval/CMSSW_3_5_2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/5E226C53-D91E-DF11-9BB6-003048678B0C.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/D8C9C5C7-1C1E-DF11-994F-003048678F26.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/8E8466CA-1C1E-DF11-9EC9-001A928116D4.root',
        '/store/relval/CMSSW_3_5_2/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0015/3A6190AA-1B1E-DF11-A354-002618943957.root'


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



