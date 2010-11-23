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
from Validation.RecoEgamma.conversionPostprocessing_cfi import *

photonValidation.OutputFileName = 'PhotonValidationRelVal392_H130GGgluonfusion_FastSim.root'
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

        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0073/12994C21-A8E9-DF11-A178-001A928116C2.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/E49CB54D-11E8-DF11-A4EB-0026189438EB.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/DCAF69E2-12E8-DF11-94DA-0018F3D096BC.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/D0FF0A65-14E8-DF11-8D27-0026189438FF.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/B096A94D-11E8-DF11-8DD5-00248C0BE005.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/AA786CEF-11E8-DF11-A9C5-001A92971B88.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/A251DC53-12E8-DF11-B2DA-001A9281173A.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/86D9DFE9-11E8-DF11-86D3-001A92810ACE.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/865BC9EC-11E8-DF11-82B8-001A92971BB4.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/74B1C8EF-11E8-DF11-8894-001A9281170E.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/72194F4F-11E8-DF11-9667-002618943956.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/6820D04E-11E8-DF11-8A29-0026189437F8.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/4E1BE7EC-11E8-DF11-A03F-001A92811706.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V3_FastSim-v1/0067/2C34F1E6-11E8-DF11-8BDD-003048678FF6.root'

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
