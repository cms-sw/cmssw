
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
process.GlobalTag.globaltag = 'START38_V8::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre1_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre1 QCD_Pt_80_120

        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0009/5A3B6A4A-079B-DF11-B2C8-0018F3D09654.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0008/F417D4D1-E59A-DF11-AE6F-0018F3D0962C.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0008/8800745A-E19A-DF11-B9F6-002618943852.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0008/58DC1270-E09A-DF11-9CC8-00304867C0F6.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0008/52A139D9-E19A-DF11-A654-002618943945.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0008/3E2E6359-E79A-DF11-8A62-0018F3D09704.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0008/2C674E5C-E19A-DF11-9E93-00261894385A.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V8-v1/0008/2A7DC4E1-E09A-DF11-AFEA-003048678FC4.root'
 
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre1 QCD_Pt_80_120
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0009/68B8F943-079B-DF11-856F-00248C55CC9D.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/FC17FC6F-E09A-DF11-88A2-002354EF3BE1.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/F206BD6F-E09A-DF11-8458-00304867C0C4.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/EA96D352-E19A-DF11-BF5C-0026189438AB.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/C22CF5D6-E69A-DF11-AB40-00304867C1BC.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/B413D04B-E69A-DF11-8294-003048679048.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/AA6E9363-E79A-DF11-8D6B-002618943980.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/A4EF1054-E59A-DF11-A32D-001A92971BC8.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/9EDECAD7-E19A-DF11-B9A7-00261894387A.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/9CCCB66F-E09A-DF11-8E24-0026189438E3.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/8C9A8DD9-E19A-DF11-BB77-0026189438A5.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/7CADF852-E19A-DF11-97B0-0026189438ED.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/72D12968-E09A-DF11-AE38-00261894394F.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/642124CF-E29A-DF11-A10D-002618FDA216.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/4EC51070-E09A-DF11-8420-003048679166.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/2233E3DD-E09A-DF11-8FEE-0030486792B8.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/1EF61A5A-E19A-DF11-BB58-002618943926.root'

         
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


