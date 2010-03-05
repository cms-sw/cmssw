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
photonValidation.OutputFileName = 'PhotonValidationRelVal350_SingleGammaPt35.root'
photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 350 single Photons pt=35GeV            
        '/store/relval/CMSSW_3_5_0/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V21-v1/0013/F60A5292-6213-DF11-98ED-001A92810ADE.root',
        '/store/relval/CMSSW_3_5_0/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V21-v1/0013/30B7AD43-4713-DF11-9901-0018F3D0962E.root'

    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(
    
# official RelVal 350 single Photons pt=35GeV

        '/store/relval/CMSSW_3_5_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0013/AE76CCC9-4613-DF11-801D-003048678CA2.root',
        '/store/relval/CMSSW_3_5_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0013/5EA47F42-4713-DF11-9672-002618FDA265.root',
        '/store/relval/CMSSW_3_5_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0013/42E85A20-4813-DF11-A941-0017319C92DA.root',
        '/store/relval/CMSSW_3_5_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0013/0CB18390-6213-DF11-9795-0018F3D095FA.root'

    )
 )


photonPostprocessing.rBin = 48

## For single gamma pt = 35
photonValidation.eMax  = 300
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonValidation.dCotCutOn = False
photonValidation.dCotCutValue = 0.15

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)



process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
