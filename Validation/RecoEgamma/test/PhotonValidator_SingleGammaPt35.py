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
process.GlobalTag.globaltag = 'MC_36Y_V7A::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal361_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 361 single Photons pt=35GeV            
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_36Y_V7A-v1/0021/320F0F9D-515D-DF11-9C35-001A928116E2.root',
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_36Y_V7A-v1/0020/72582EA2-175D-DF11-BAEE-00261894398C.root',
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_36Y_V7A-v1/0020/5237BDAF-165D-DF11-AEDB-001A92971B92.root' 

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 361 single Photons pt=35GeV
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V7A-v1/0021/54F81996-515D-DF11-8D82-002618943977.root',
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V7A-v1/0020/DE9EC1A2-175D-DF11-8ADB-002618943831.root',
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V7A-v1/0020/C088F79C-165D-DF11-A6B6-0026189438F3.root',
        '/store/relval/CMSSW_3_6_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V7A-v1/0020/226C2F9D-165D-DF11-84CE-001A92811748.root'
        
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
