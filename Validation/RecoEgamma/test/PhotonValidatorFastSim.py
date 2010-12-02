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
process.load("Validation.RecoEgamma.conversionPostprocessing_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START39_V5::All'

process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *
from Validation.RecoEgamma.conversionPostprocessing_cfi import *


photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre5_H130GGgluonfusion_FastSim.root'
photonValidation.fastSim = True
photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName
photonPostprocessing.OuputFileName = photonValidation.OutputFileName

#conversionPostprocessing.standalone = cms.bool(True)
#conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName
#conversionPostprocessing.OuputFileName = tkConversionValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0090/E627200A-91F5-DF11-B518-001A92811732.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/EEB450D0-C4F4-DF11-8B11-001A928116D2.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/D4682A64-C2F4-DF11-9C71-003048678B72.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/CC05481B-C7F4-DF11-9FBD-003048678E94.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/BC471A71-C2F4-DF11-9873-002618943944.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/B89117F6-C2F4-DF11-A6FD-001A92971AAA.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/ACA2A9CB-C9F4-DF11-974F-00304867BED8.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/A8B29949-C8F4-DF11-B2D5-00261894396F.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/82C6AB68-C2F4-DF11-BB27-001A928116DC.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/48FEBC66-C4F4-DF11-8F8D-001A92810A9E.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/383FCA70-C2F4-DF11-A501-0018F3D09624.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/324F0E5F-C2F4-DF11-91D6-0018F3D0965A.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/22177C0D-C6F4-DF11-87B0-003048678B72.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V5_FastSim-v1/0084/0C4C3A75-C2F4-DF11-8F00-003048678CA2.root'

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




#process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.p1 = cms.Path(process.tpSelection*process.photonValidation*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
