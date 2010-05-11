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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre3_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre3 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V2-v1/0019/6293B21E-F157-DF11-BE03-001A928116C2.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V2-v1/0019/3207B279-3458-DF11-984A-0026189438E0.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V2-v1/0018/EE7B1EB6-EC57-DF11-B4CA-0018F3D0960E.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V2-v1/0018/647240DF-E957-DF11-A1AE-0026189438A2.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V2-v1/0018/52427A90-EB57-DF11-B446-001A92971AA8.root'
    ),


    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre3 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/E6D8C15E-0F58-DF11-860C-0018F3D09616.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/40D6B572-F657-DF11-A264-0018F3D0965E.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/E617F16B-E957-DF11-9959-001BFCDBD176.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/AC2003F9-E457-DF11-95EC-003048678AC0.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/7CAD748D-EB57-DF11-9CF0-003048678FDE.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/6CD6938F-ED57-DF11-A40D-0018F3D0960E.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/5699CBEE-EA57-DF11-A5C2-0026189438AE.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/36204026-EC57-DF11-90D6-00261894387C.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/2ECF860B-EC57-DF11-934A-002618FDA237.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0018/22B825B2-EC57-DF11-BA95-00261894386F.root'


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
