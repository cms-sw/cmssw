
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
process.GlobalTag.globaltag = 'START37_V2::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre3_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre3 QCD_Pt_80_120
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V2-v1/0019/E20D9D4F-3558-DF11-B2CA-0018F3D0967E.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V2-v1/0018/F87102EE-E957-DF11-8DB5-0018F3D0968C.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V2-v1/0018/A4BCBF03-EE57-DF11-B565-0018F3D09654.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V2-v1/0018/8853BB6F-E857-DF11-AF0D-001A928116DE.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V2-v1/0018/46B07501-EB57-DF11-B36F-0018F3D09706.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V2-v1/0018/3406676D-EA57-DF11-BD82-00304867916E.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V2-v1/0018/12F8B646-E757-DF11-9021-0018F3D095F8.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V2-v1/0018/0A17D1CE-E757-DF11-A1A4-001A92971B8C.root'
        
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre3 QCD_Pt_80_120
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/526B7E5E-0E58-DF11-B83C-0018F3D096A6.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/2CB2CD30-3558-DF11-B7DC-0018F3D0967E.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/D29D6E6E-E857-DF11-B35D-0018F3D0968E.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/C2C74369-EA57-DF11-A24D-002618943926.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/BC73536F-E857-DF11-B4DC-001A928116E6.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/B4DE85F4-EB57-DF11-9882-003048678B36.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/B09F5FEC-E957-DF11-86BD-0018F3D09664.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/AAB753D8-E857-DF11-84F0-0018F3D095F8.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/924F71BF-E657-DF11-B637-0018F3D096CE.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/760800EB-E957-DF11-B20D-001A928116FA.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/6C9D2B6D-E857-DF11-B839-003048679244.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/682A4164-E957-DF11-ADBB-001A92971B68.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/5C9CE6BF-E657-DF11-B665-001A928116D0.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/384330FE-EA57-DF11-A449-0018F3D0963C.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/366D22EF-E957-DF11-93AA-0018F3D0961A.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/1CD9CACE-E757-DF11-88F4-001A92810AE0.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/1223DA75-EC57-DF11-9338-0018F3D096DA.root'

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


