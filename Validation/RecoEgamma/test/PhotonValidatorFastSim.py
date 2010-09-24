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
#process.GlobalTag.globaltag = 'MC_38Y_V8::All'
process.GlobalTag.globaltag = 'START38_V11::All'

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
#photonValidation.OutputFileName = 'PhotonValidationRelVal390pre3_SingleGammaPt10_FastSim.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre4_H130GGgluonfusion_FastSim.root'

photonValidation.fastSim = True

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0027/1018DDD6-74C3-DF11-B7BE-003048678F26.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0026/C23C7679-2BC3-DF11-BD35-0026189438AB.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0026/925E5F7F-2CC3-DF11-A2F0-002618FDA259.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0026/14CC2F83-2CC3-DF11-BE31-0018F3D09608.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0025/9E3E9A88-29C3-DF11-B6FB-0018F3D09670.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0025/9689ADFB-28C3-DF11-8EAB-0018F3D09670.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0025/942B3489-29C3-DF11-BBC2-001A92810AC0.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0025/60011E8E-2AC3-DF11-98F7-002618943944.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0025/56285305-2AC3-DF11-A0ED-0018F3D09644.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0025/3ADA4201-2BC3-DF11-B40D-0018F3D095F0.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0025/28ABC68C-2AC3-DF11-94E4-003048678B34.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0025/24A3E901-2BC3-DF11-8E9B-001BFCDBD1BE.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0025/106B7707-2AC3-DF11-96E3-0018F3D09616.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V11_FastSim-v1/0025/1065C591-2AC3-DF11-B252-001A9281172C.root'




    )
    )


photonPostprocessing.rBin = 48

## For single gamma pt =10
#photonValidation.eMax  = 100
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonPostprocessing.eMax  = 100
#photonPostprocessing.etMax = 50

## For single gamma pt = 35
#photonValidation.eMax  = 300
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonValidation.dCotCutOn = False
#photonValidation.dCotCutValue = 0.15


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
