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
process.GlobalTag.globaltag = 'START38_V12::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre4_PhotonJets_Pt_10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre4 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V11-v1/0026/7679CC3C-67C3-DF11-B3F5-002618FDA277.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V11-v1/0025/DCCA8FED-0EC3-DF11-895F-003048678F92.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V11-v1/0025/5AE4B65C-13C3-DF11-A589-00304867920C.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V11-v1/0025/52AF4865-15C3-DF11-B82D-0018F3D096C0.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V11-v1/0025/163BF065-0EC3-DF11-9D53-0018F3D09670.root'
    ),


    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre4 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0027/34509BDA-74C3-DF11-9A4F-002618943957.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/CC0416E8-0EC3-DF11-B979-001A92971AAA.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/C6FD7EE0-0FC3-DF11-85DA-001A9281171E.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/AA2AA565-0DC3-DF11-9199-00261894388D.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/806CA363-0DC3-DF11-954A-003048679006.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/48F40A5B-13C3-DF11-A2E9-00304867916E.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/28B15DE1-15C3-DF11-A87A-0018F3D095FC.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/165343E0-0FC3-DF11-8712-003048678F26.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/14981DDD-13C3-DF11-A7F3-0018F3D096DA.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/0CEFF763-0EC3-DF11-B12C-003048678F92.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/0217345E-14C3-DF11-BBDB-003048678F74.root'

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
