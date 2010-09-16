
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
photonValidation.OutputFileName = 'PhotonValidationRelVal383_QCD_Pt_80_120.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 383 QCD_Pt_80_120
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0022/46618384-E1BF-DF11-A1AB-00304867C136.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0021/E803E766-9DBF-DF11-9E34-0018F3D095F2.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0021/C07F94D6-9FBF-DF11-A098-002618943949.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0021/949009E5-9BBF-DF11-BD41-002618FDA207.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0021/8CD992E1-9DBF-DF11-A8D5-003048678BAE.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0021/60B76770-9CBF-DF11-99C5-001A92810AF4.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0021/38C6BE72-9DBF-DF11-BFA3-0018F3D096F8.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0021/10952870-9EBF-DF11-9715-0018F3D096A2.root'
 
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 383 QCD_Pt_80_120
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/2EC5D590-E1BF-DF11-9711-002354EF3BDF.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/DAFED5EA-9CBF-DF11-A4D2-0018F3D09708.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/D81180ED-9DBF-DF11-A5A4-0018F3D09676.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/C84FFA71-9EBF-DF11-AA54-0018F3D096DC.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/C0F46666-A0BF-DF11-AEB3-0018F3D095F2.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/BA327BF0-9CBF-DF11-B84C-0018F3D096DE.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/9CEF3672-9DBF-DF11-B388-001A92810AF4.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/8E38736F-9CBF-DF11-9425-0018F3D09692.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/8CF3ADDE-9DBF-DF11-A58F-0018F3D0965E.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/84DCD36A-9CBF-DF11-83A4-002618FDA28E.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/7438EE6D-9CBF-DF11-8D6C-0018F3D096E0.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/58AE46E4-9BBF-DF11-9AA8-0026189437F5.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/52254361-9DBF-DF11-893F-003048678A7E.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/3A02AA71-9DBF-DF11-A432-0018F3D09660.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/2C97A1D7-9FBF-DF11-98CA-001A9281172C.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/263FDA71-9EBF-DF11-B34E-0018F3D096D4.root',
        '/store/relval/CMSSW_3_8_3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/1446796B-9DBF-DF11-961A-0018F3D09676.root'
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


