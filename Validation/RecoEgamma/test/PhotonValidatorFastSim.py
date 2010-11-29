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
process.GlobalTag.globaltag = 'START39_V6::All'

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

photonValidation.OutputFileName = 'PhotonValidationRelVal394_H130GGgluonfusion_FastSim.root'
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
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0001/52D1563B-34F8-DF11-9842-002618943885.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/F218186F-F3F7-DF11-B7C9-003048D15DB6.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/D2DBF9DF-DFF7-DF11-96C9-00248C55CC4D.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/BA513BBB-0FF8-DF11-B34C-0018F3D09604.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/B43AB03B-0FF8-DF11-B920-00261894395C.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/A435BD2A-0FF8-DF11-92F1-0018F3D09604.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/902A0327-0FF8-DF11-9DB4-002354EF3BE3.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/8C98342C-0FF8-DF11-A2D7-0018F3D096BE.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/76C444E1-DEF7-DF11-8546-00304867BFF2.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/6EEDFD72-F5F7-DF11-814F-0026189438EF.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/6274F4BE-0FF8-DF11-9F5A-002618943877.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/3439000B-DBF7-DF11-8D01-002618FDA263.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/222F42D4-DCF7-DF11-8DA6-003048678E92.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START39_V6_FastSim-v1/0000/1260D2EC-DBF7-DF11-AC96-001A9281172A.root'

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
