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
process.GlobalTag.globaltag = 'START38_V6::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre8_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre8 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V6-v1/0001/446924E4-A58B-DF11-83F1-002618943905.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V6-v1/0000/F20E2035-608B-DF11-8A66-00261894386D.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V6-v1/0000/9E00324A-5F8B-DF11-96D9-001A928116FA.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V6-v1/0000/766C4FDD-608B-DF11-A05F-001A92810A96.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V6-v1/0000/4E3642A9-5A8B-DF11-8657-0026189438F9.root' 
    ),


    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre8 RelValPhotonJets_Pt_10
  '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/565021B7-A58B-DF11-A2B4-001BFCDBD100.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/FCD81C9D-5E8B-DF11-9402-002618943937.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/DCC60F73-598B-DF11-8984-002618943914.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/D6665B3C-5F8B-DF11-A6C2-001A928116DE.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/BC07B8A6-618B-DF11-A8F7-001A92811716.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/9A12DF51-608B-DF11-9891-0018F3D09600.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/6CF11E61-608B-DF11-9601-001A92811726.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/628D3826-5E8B-DF11-9227-001A92811700.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/3C895C7B-5A8B-DF11-8EC1-003048678B3C.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/2A37AC75-608B-DF11-B95B-001A92971BCA.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/28B4218B-5F8B-DF11-B3AD-00261894393C.root'

    )
 )


photonPostprocessing.rBin = 48
## For gam Jet and higgs
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
