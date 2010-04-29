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
process.GlobalTag.globaltag = 'MC_37Y_V0::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre1_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre1 single Photons pt=35GeV
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V0-v1/0015/DC62B367-0E4D-DF11-BAD7-002618943961.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V0-v1/0014/8C098D1E-B94C-DF11-A837-0018F3D0970A.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V0-v1/0014/52F73C74-BE4C-DF11-BA1B-001A928116BE.root'
      
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre1 single Photons pt=35GeV
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0015/5AB5E25D-0E4D-DF11-9EE1-00261894397D.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0014/A6F9B95B-BD4C-DF11-AACB-003048678B0E.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0014/4AE78717-B64C-DF11-A373-001A92810A96.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V0-v1/0014/3646822E-B94C-DF11-B802-00261894395A.root'

      
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
