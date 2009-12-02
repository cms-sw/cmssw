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
process.GlobalTag.globaltag = 'MC_31X_V9::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal335_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

# official RelVal 335 single Photons pt=35GeV            

        '/store/relval/CMSSW_3_3_5/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V9-v1/0008/B4B8348F-12DC-DE11-BB4B-0018F3D0963C.root',
        '/store/relval/CMSSW_3_3_5/RelValSingleGammaPt35/GEN-SIM-RECO/MC_31X_V9-v1/0007/2CBBB8BF-CADB-DE11-B6E8-002618943894.root'
  
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 335 single Photons pt=35GeV
        '/store/relval/CMSSW_3_3_5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/C67BB188-12DC-DE11-98CC-001731AF6BCB.root',
        '/store/relval/CMSSW_3_3_5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/F08BB5C4-CADB-DE11-AC7C-001BFCDBD184.root',
        '/store/relval/CMSSW_3_3_5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/A64AACC2-CADB-DE11-9E2D-002618943874.root',
        '/store/relval/CMSSW_3_3_5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/48EA5FBE-CADB-DE11-B7B9-002618943915.root'
     
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
