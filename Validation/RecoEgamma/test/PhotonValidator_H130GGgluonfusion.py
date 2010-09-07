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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre3_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre3 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0021/7A2E8ABB-74B6-DF11-B208-003048678FF6.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0020/CEB2E11B-30B6-DF11-8C52-0030486792BA.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0020/2C5C9BA4-22B6-DF11-8B84-0018F3D096EE.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0019/7C6769E7-06B6-DF11-9A09-002618FDA277.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0019/3816335C-F9B5-DF11-BA2B-00261894387E.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre3 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/38C359BC-74B6-DF11-987C-001A928116EC.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/E8E0FF28-26B6-DF11-A4F0-003048679044.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/E8D74F22-22B6-DF11-9519-0030486790C0.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/3C09040C-36B6-DF11-BFB7-001A92971B3C.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/363424B0-29B6-DF11-B3F4-0030486791F2.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0020/2E39390F-31B6-DF11-85AC-003048678FC4.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0019/CC199EDD-06B6-DF11-B116-00261894388B.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0019/CA593067-F8B5-DF11-8626-00261894391B.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0019/88E26500-F7B5-DF11-8132-003048678BAA.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0019/64042A55-FEB5-DF11-AF7B-002618943880.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0019/5418A9C7-0DB6-DF11-BCDE-0030486790BE.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0019/520E1041-07B6-DF11-B244-00261894386D.root'


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
