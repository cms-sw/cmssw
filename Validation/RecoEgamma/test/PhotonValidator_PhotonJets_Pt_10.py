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
process.GlobalTag.globaltag = 'START3X_V25::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal355_PhotonJets_Pt_10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 355 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START3X_V25-v1/0006/F2A525C9-0F38-DF11-BEC5-003048678B1A.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START3X_V25-v1/0006/C8322111-C437-DF11-B832-001BFCDBD100.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START3X_V25-v1/0006/AA3A7A9C-BE37-DF11-8693-002618943964.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START3X_V25-v1/0006/76195210-BE37-DF11-8BDD-0026189438DA.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START3X_V25-v1/0006/60E4C911-BD37-DF11-8F32-0026189438DD.root'
 
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 355 RelValPhotonJets_Pt_10

        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/D841FC0E-BD37-DF11-A73B-00261894386C.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/C62F2819-BF37-DF11-B5FB-0026189438DD.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/B6DFC4AB-BD37-DF11-BD48-0026189438E9.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/AE22BCAB-BD37-DF11-A66B-00261894386F.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/90CDA450-C637-DF11-B23D-0018F3D09628.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/88972C19-BE37-DF11-B777-00304867BEDE.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/84F4907E-C337-DF11-8559-001A92811708.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/78C4200E-BD37-DF11-A74F-00261894395C.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/50CD1514-BE37-DF11-A306-00261894391B.root',
        '/store/relval/CMSSW_3_5_5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/16C65A9C-BE37-DF11-BFB6-0018F3D0960A.root'
    
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
