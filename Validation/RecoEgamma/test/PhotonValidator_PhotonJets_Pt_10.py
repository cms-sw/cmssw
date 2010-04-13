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
process.GlobalTag.globaltag = 'START36_V3::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre5_PhotonJets_Pt_10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

# official RelVal 360pre5 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V3-v1/0010/CCA85081-493E-DF11-A002-002618943951.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V3-v1/0009/CE8E4FD9-D33D-DF11-ADA4-0018F3D095F2.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V3-v1/0009/C60F9CF4-D43D-DF11-B7FF-003048679220.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V3-v1/0009/B48249A0-CF3D-DF11-A4DD-0030486792BA.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V3-v1/0009/1CF90870-D03D-DF11-A6FB-00261894386F.root'
        
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360pre5 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/DEE34CD6-D33D-DF11-81C3-003048678B20.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/CE62998D-CF3D-DF11-9759-0018F3D0968C.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/A65BFC8B-CF3D-DF11-9B76-001A928116C0.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/94BF2985-CF3D-DF11-B46D-0018F3D096CE.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/90ABF1D5-D33D-DF11-92BA-002618FDA25B.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/7E1A8156-D03D-DF11-AA58-002618943900.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/7CD6D9F8-D43D-DF11-AB9B-001A92971BBE.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/36842CB1-D03D-DF11-9204-001A92971B88.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/22A223F3-D83D-DF11-ABE0-001A92811722.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/1626B683-CF3D-DF11-84EC-0030486790C0.root'
    
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
