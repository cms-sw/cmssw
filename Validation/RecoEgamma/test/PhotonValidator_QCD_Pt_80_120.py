
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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre3_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre3 QCD_Pt_80_120
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0021/1CF0F6B8-74B6-DF11-9914-001A928116F0.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0020/EA12E5C3-21B6-DF11-A394-002618943884.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0020/D8BDAEB8-33B6-DF11-9007-003048678FA0.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0020/CEA2D9C4-21B6-DF11-A2C5-0026189438FA.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0020/7E090D2B-30B6-DF11-814D-0018F3D096F6.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0020/5239E220-2AB6-DF11-8EB6-002618943964.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0020/1A86201F-24B6-DF11-A0D1-0018F3C3E3A6.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0020/18D46C02-3FB6-DF11-AD8A-001A92810AD2.root'
 
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre3 QCD_Pt_80_120

        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/5A0FB5B6-74B6-DF11-86EE-001BFCDBD11E.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/DCEDAC91-32B6-DF11-9CD1-00304867926C.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/B4BAF67A-3CB6-DF11-B9C7-001A92810A9E.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/B099A1C4-21B6-DF11-95E5-0018F3D09664.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/AEE523C1-21B6-DF11-BDAD-003048678E2A.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/90408A25-2EB6-DF11-BB0C-003048679076.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/8AA57BC3-21B6-DF11-992C-002618943967.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/7CD4331D-30B6-DF11-9AD6-001BFCDBD1BC.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/7C0BBFB3-36B6-DF11-B469-003048678E94.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/6AD344B0-1CB6-DF11-BCE6-003048678AE2.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/60EA1D38-2CB6-DF11-A68E-0018F3D0963C.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/608F1C61-41B6-DF11-91F7-001A92971B36.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/2ABDCDA9-27B6-DF11-AAB1-003048678C06.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/282DB121-31B6-DF11-8304-0030486792BA.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/16D685C5-21B6-DF11-A124-001A928116F4.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/0E59762B-28B6-DF11-B4BC-0030486792B6.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/0CEC4D3C-22B6-DF11-92C8-0018F3D09664.root'

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


