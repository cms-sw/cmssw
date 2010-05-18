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
process.GlobalTag.globaltag = 'START37_V3::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre4_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre4 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V3-v1/0022/4E3AA784-A85D-DF11-9E76-0026189437F9.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V3-v1/0021/FE5A32F4-785D-DF11-9179-001A92810AB2.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V3-v1/0021/82F16374-7A5D-DF11-9181-001A9281172C.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V3-v1/0021/3690ED5E-775D-DF11-B4C0-001A92971ACC.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V3-v1/0021/1EB42DAB-7F5D-DF11-A92E-001A92810ACA.root'
    ),


    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre4 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0022/8C073532-A85D-DF11-A853-002618FDA250.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/FE9BAE12-7F5D-DF11-95A0-001A92811744.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/FCC4D2F0-785D-DF11-8351-001A92810AF4.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/F65EE15D-785D-DF11-B29F-0018F3D09600.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/C60974D9-765D-DF11-B00E-003048679228.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/AA5FD4A9-7F5D-DF11-930C-001A92811736.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/A6C6416C-795D-DF11-9C2A-001A92971B32.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/A641D6C4-755D-DF11-A24B-0026189437F5.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/809A4CF9-795D-DF11-9AF2-001A92810AEA.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/64C95E31-7C5D-DF11-87C5-001A92811736.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/406E6232-835D-DF11-9660-001A9281170E.root'


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
