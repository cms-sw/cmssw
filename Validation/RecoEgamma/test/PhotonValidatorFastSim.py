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
#photonValidation.OutputFileName = 'PhotonValidationRelVal390pre3_SingleGammaPt10_FastSim.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre3_H130GGgluonfusion_FastSim.root'
photonValidation.fastSim = True

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0021/D4CBB7AA-74B6-DF11-98F1-0026189438A2.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/EC81CAF1-FCB5-DF11-A9A3-00261894397B.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/EAC69757-FAB5-DF11-8898-0026189438CF.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/CCFE8BEC-FCB5-DF11-993C-002618943904.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/CC8A2F55-FBB5-DF11-BA44-002618943899.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/8E488C56-FAB5-DF11-BDAA-003048678E94.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/7668F953-FBB5-DF11-BF40-002618FDA262.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/64C03054-FBB5-DF11-A889-002618943875.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/64A69456-FAB5-DF11-9291-0030486792A8.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/2E461D55-FAB5-DF11-A902-0026189438DC.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/2C1030F0-FCB5-DF11-A120-001A9281172E.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/26291BED-FCB5-DF11-B918-0026189438E6.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/183F9F54-FAB5-DF11-BAF2-003048678A80.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START38_V9_FastSim-v1/0019/02AA2C59-FAB5-DF11-884E-00261894398B.root'    


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
