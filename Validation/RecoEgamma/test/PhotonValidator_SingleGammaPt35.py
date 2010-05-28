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
process.GlobalTag.globaltag = 'MC_37Y_V4::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre5_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre5 single Photons pt=35GeV
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V4-v1/0023/7ADD6C54-8963-DF11-AD30-003048678FC4.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V4-v1/0022/92C25FEF-4E63-DF11-AEDB-00261894392D.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V4-v1/0022/14BC177A-4F63-DF11-BC82-00261894392D.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre4 single Photons pt=35GeV
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0023/E86BB64F-8963-DF11-9BAE-002618943933.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0022/AED138F5-4E63-DF11-A828-0026189438E0.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0022/54BA4176-4F63-DF11-9D94-0026189438CF.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0022/06A5EB62-4E63-DF11-8B55-002618943882.root'

        
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
