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
process.GlobalTag.globaltag = 'START3X_V26::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal357_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 357 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START3X_V26-v1/0012/C03E69F0-6849-DF11-A609-0018F3D09678.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START3X_V26-v1/0012/C024FE3B-4349-DF11-8DE1-003048D3FC94.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START3X_V26-v1/0012/8CEA2632-4249-DF11-B144-003048D15D04.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START3X_V26-v1/0012/1E7C30B9-4249-DF11-9596-003048678F6C.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START3X_V26-v1/0012/14EF3BD0-4349-DF11-8CCA-002354EF3BDA.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 357 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/CCE722A9-4149-DF11-BFA3-003048678B84.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/CC3E073A-4349-DF11-A5A2-003048679010.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/C8266D44-4449-DF11-83D3-003048679166.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/C6FEDA38-4349-DF11-BA93-003048678B18.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/5EAA97B6-4249-DF11-929A-003048678B00.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/36C14434-4249-DF11-9270-0018F3D096B4.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/36704ACE-4349-DF11-99E1-003048679296.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/280A1B32-4249-DF11-965C-003048D42D92.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/20705FB7-4249-DF11-9474-003048D15CC0.root',
        '/store/relval/CMSSW_3_5_7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/10BF6CBB-4249-DF11-B6C1-003048678FEA.root'   
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
