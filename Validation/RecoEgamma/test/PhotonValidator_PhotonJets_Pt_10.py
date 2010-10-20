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
process.GlobalTag.globaltag = 'START39_V2::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal390_PhotonJets_Pt_10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V2-v1/0052/604F7588-4DD8-DF11-BC2E-0018F3D09700.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V2-v1/0050/D81F5D02-FED7-DF11-8FC4-00261894390C.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V2-v1/0050/9888B365-FDD7-DF11-92F2-0018F3D09634.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V2-v1/0050/8A303228-04D8-DF11-86F4-0018F3D09644.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V2-v1/0050/0C4BA95D-FCD7-DF11-B70D-001A928116DA.root'
    ),


    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/EEDD6760-FDD7-DF11-95B4-002618943913.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/EA059B09-FDD7-DF11-8F39-0026189437EB.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/D20E180A-FDD7-DF11-AFA4-00261894397F.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/BC4B2700-FED7-DF11-9EE7-002618943915.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/A85D390A-FDD7-DF11-88D3-0026189438FF.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/9ED1F75A-FCD7-DF11-8788-003048678B8E.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/6CF9C905-FDD7-DF11-B748-002618943915.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/680D4106-FDD7-DF11-A7D1-0026189438A9.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/0A863306-04D8-DF11-9670-0018F3D096E4.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/00C176BB-39D8-DF11-B9D8-001A92971B12.root',
        '/store/relval/CMSSW_3_9_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/00AB3306-02D8-DF11-9ABB-0018F3D09650.root'



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
