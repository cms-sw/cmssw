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
process.GlobalTag.globaltag = 'START39_V3::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre2_H130GGgluonfusion_FastSim.root'

photonValidation.fastSim = True

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0058/168E0A6C-DFE2-DF11-8C1C-002618943896.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/E8EF2EBE-80E2-DF11-BD5D-00261894389A.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/DA005739-80E2-DF11-BE4C-003048678EE2.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/C64400A9-82E2-DF11-A6FA-002618943954.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/B6746E1D-80E2-DF11-B2BF-0026189438E4.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/A8BBA4BC-80E2-DF11-8859-0026189438D4.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/920921C2-80E2-DF11-BE5D-002618943860.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/680681C1-80E2-DF11-94E2-003048678F78.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/58165CC3-80E2-DF11-A3CB-0018F3D0965C.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/524D9A40-80E2-DF11-87BD-001A92810AAE.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/464F8C2B-80E2-DF11-AE25-00304867904E.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/42B4021D-80E2-DF11-A19D-001A92811702.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/1ECC5CBA-80E2-DF11-9EB8-00248C55CC62.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0057/0699064D-80E2-DF11-8E58-0018F3D0965C.root'

 

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
