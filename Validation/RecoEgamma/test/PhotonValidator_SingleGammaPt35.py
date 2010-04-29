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
process.GlobalTag.globaltag = 'MC_37Y_V1::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre2_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre2 single Photons pt=35GeV
        '/store/relval/CMSSW_3_7_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V1-v1/0018/30520DB9-F652-DF11-BF04-00248C55CC9D.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V1-v1/0017/F88DA277-8B52-DF11-9617-00261894393C.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V1-v1/0017/E23BBE2D-8B52-DF11-B1DC-0018F3D0967A.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre2 single Photons pt=35GeV
        '/store/relval/CMSSW_3_7_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V1-v1/0018/A815C312-F752-DF11-906E-0026189438CF.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V1-v1/0017/EEFD9C76-8B52-DF11-85AB-002618943935.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V1-v1/0017/5AE5F62D-8B52-DF11-BF45-002618943978.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V1-v1/0016/9A6DAA71-8A52-DF11-B2DB-003048678BEA.root' 
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
