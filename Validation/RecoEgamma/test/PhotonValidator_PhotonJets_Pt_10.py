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
process.GlobalTag.globaltag = 'START38_V9::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre3_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre3 RelValPhotonJets_Pt_10

        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0021/2876FEBA-74B6-DF11-8E74-003048678B1A.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0020/C83A8285-35B6-DF11-A70B-0030486792BA.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0020/7C52AFAB-25B6-DF11-BCC2-0030486792AC.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0020/4CAE6DD6-15B6-DF11-B4B1-001A92810AE0.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V9-v1/0019/CE74A8BC-10B6-DF11-A8D2-003048D3C010.root'
  
    ),


    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre3 RelValPhotonJets_Pt_10

        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/F43C39BB-74B6-DF11-B746-0026189438B3.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/E2CDB2CC-15B6-DF11-BD8A-003048D15DDA.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/DEF39690-32B6-DF11-B3DB-003048678D86.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/AAADA6B1-12B6-DF11-8A0D-003048678B20.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/9C37EC22-30B6-DF11-8672-003048679084.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/5EF274B9-25B6-DF11-A10B-0018F3D0964A.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/4AF35DDD-1CB6-DF11-8913-0026189437ED.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/228094B8-14B6-DF11-8C96-0018F3D096A6.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/12B50A6D-3DB6-DF11-8868-001A92971B8C.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0019/12EAA6B9-10B6-DF11-8C32-002618943860.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0019/06285A39-10B6-DF11-9517-001A92971B28.root'
        
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
