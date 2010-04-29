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
process.GlobalTag.globaltag = 'START37_V0::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre1_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre1 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V0-v1/0015/B6A14A62-0E4D-DF11-BF6E-0026189438FC.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V0-v1/0014/F4C9C2A5-C94C-DF11-A0E5-0018F3D096AA.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V0-v1/0014/E284F6B5-CD4C-DF11-BD56-00248C55CC3C.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V0-v1/0014/D677F4BB-C94C-DF11-AA85-003048678B3C.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V0-v1/0014/04ACE389-C84C-DF11-BE54-0030486792DE.root'      
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre1 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/3027BBFD-DA4C-DF11-8E46-003048678B0E.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0014/FC5DAA9F-C94C-DF11-8FD5-001A928116CC.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0014/F013A7F4-C84C-DF11-869E-0026189438AE.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0014/900F02F4-C84C-DF11-BC5B-00261894389E.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0014/7022EEF7-C74C-DF11-8DDE-0026189438E7.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0014/608E0570-C84C-DF11-8DBE-001A92811732.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0014/5E5F140E-CA4C-DF11-A26B-0018F3D09658.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0014/5CE58BA4-C94C-DF11-BB5D-001A928116DC.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0014/2E826AAC-C94C-DF11-BC3C-0018F3D096DA.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0014/2C260A3E-CE4C-DF11-B6B9-002354EF3BDE.root' 
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
