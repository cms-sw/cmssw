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
process.GlobalTag.globaltag = 'START38_V12::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre4_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre4 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V11-v1/0027/38BAA444-75C3-DF11-920E-0018F3D096DA.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V11-v1/0025/D4856D62-1DC3-DF11-880F-001A92810AA4.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V11-v1/0025/CCDCFBEE-17C3-DF11-AB3D-001A92810AC8.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V11-v1/0025/B45502DD-1CC3-DF11-8C15-002618943898.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V11-v1/0025/569A325F-1FC3-DF11-AF77-002618943939.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V11-v1/0025/147A8D60-16C3-DF11-BC26-002618943854.root'

    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre4 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0027/5AB09644-75C3-DF11-9497-001A92971BB8.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/EEE7E05D-1EC3-DF11-9F91-001A928116EA.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/D63F24DC-1CC3-DF11-A0BC-002618943856.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/C80947DC-15C3-DF11-B383-003048678E24.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/C033EAE2-17C3-DF11-AB67-001A92810ACE.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/B6C46B67-17C3-DF11-B65F-002618943916.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/B26599E5-1EC3-DF11-8D04-0018F3D09706.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/AE8434E1-17C3-DF11-A47B-0018F3D09658.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/A8F8C85B-18C3-DF11-965D-001A928116C8.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/9E646766-17C3-DF11-9B94-003048678F0C.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/7C0ECF61-1DC3-DF11-8B7D-0018F3D09706.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/504D6860-1DC3-DF11-9135-0026189438D3.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/2E2AB55F-16C3-DF11-9FED-0026189438CC.root'

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
