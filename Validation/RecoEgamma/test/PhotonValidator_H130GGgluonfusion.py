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
input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *

photonValidation.OutputMEsInRootFile = True
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre2_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre2 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0015/441ACE08-73A8-DF11-A6CB-002618943869.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0014/9C2F590A-24A8-DF11-A1EC-001A92971B0E.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0014/7E66FD2F-0AA8-DF11-ABE1-001A92971BB2.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0014/7AAC7FB1-0FA8-DF11-BE99-001A92971B96.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0014/56AFFCFD-1BA8-DF11-964B-003048678FC4.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0014/2C00A610-16A8-DF11-80E2-001A928116D0.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre2 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0015/CE9D340D-73A8-DF11-AFCA-0026189438D3.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/F852BE9A-1DA8-DF11-84B6-003048D15E02.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/F2C0932F-0AA8-DF11-BA0B-0018F3D09678.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/F28C362B-0EA8-DF11-85DD-001BFCDBD100.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/C48DBAAA-0FA8-DF11-AE63-0030486792BA.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/B6936323-0FA8-DF11-800D-001BFCDBD11E.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/9E9F8D94-16A8-DF11-B724-0018F3D09650.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/58FD5709-24A8-DF11-AB85-001A92971B90.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/4AC4EB18-09A8-DF11-A766-003048678BC6.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/3263E2AD-0FA8-DF11-A814-003048679012.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/2AA07D8E-18A8-DF11-AB66-003048678C26.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/20759A84-1CA8-DF11-8602-003048678B30.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/1A61D20D-16A8-DF11-BC7F-003048679236.root' 

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

#process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
