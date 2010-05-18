
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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre4_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre4 QCD_Pt_80_120
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V3-v1/0022/043E3D80-A85D-DF11-8080-00261894387C.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V3-v1/0021/B275F151-835D-DF11-8C32-001A928116DA.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V3-v1/0021/8476AE34-845D-DF11-8ED1-001A928116DA.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V3-v1/0021/48DAF5D3-7F5D-DF11-A5BC-0018F3D096CA.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V3-v1/0021/48ABB317-7B5D-DF11-8510-001A92971AD0.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V3-v1/0021/48473BB5-825D-DF11-9C1B-0018F3D096CE.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V3-v1/0021/3C201CD1-815D-DF11-B21B-0018F3D09706.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V3-v1/0021/28B46A8D-7E5D-DF11-ADC7-0030486792B4.root'        
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre4 QCD_Pt_80_120
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0022/3E45E6C7-A85D-DF11-B829-0026189438B9.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/F4A78515-7B5D-DF11-801B-003048678F74.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/F2B68F50-835D-DF11-B723-0018F3D09706.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/EC07C055-855D-DF11-9080-0018F3D09616.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/E279B6B3-825D-DF11-846D-001A92811736.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/BCB5FD50-835D-DF11-9BA7-001A9281172C.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/B6DE5C51-835D-DF11-9D6F-001BFCDBD1B6.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/A4BEAC7F-7D5D-DF11-BA12-001A92971ACC.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/9E8123B2-805D-DF11-9056-002618943831.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/9E484EE5-835D-DF11-9BA9-001A9281170E.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/9CF67A3E-7F5D-DF11-BB2A-001A92971ACC.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/82842C51-835D-DF11-95C0-001A92971BBA.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/80E229D3-7F5D-DF11-939E-001A928116DA.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/7E09E674-7A5D-DF11-A25E-001A92971B92.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/56D1813E-7F5D-DF11-9274-0018F3D09612.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/1A5A6826-825D-DF11-AA27-0018F3D09630.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/1474C900-7E5D-DF11-894F-001A928116D2.root'


        )
 )


photonPostprocessing.rBin = 48
## For gam Jet and higgs
photonValidation.eMax  = 500
photonValidation.etMax = 500
photonPostprocessing.eMax  = 500
photonPostprocessing.etMax = 500




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)


process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)


