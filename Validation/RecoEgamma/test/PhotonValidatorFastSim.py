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
process.GlobalTag.globaltag = 'START310_V1::All'

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


photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre7_H130GGgluonfusion_FastSim.root'
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
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0102/F226C829-39FD-DF11-9C82-001A92810AD4.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/EA60BBDD-D3FC-DF11-B314-003048678A7E.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/DE0965DD-D3FC-DF11-89A0-003048679000.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/CAA96167-D5FC-DF11-90F1-00304867BECC.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/B6889FDD-D3FC-DF11-AC7F-003048678D78.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/AC4513E2-D5FC-DF11-9D72-00261894386A.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/60C79BD5-D3FC-DF11-A6F2-0026189438E9.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/5C082D69-D6FC-DF11-AA87-001A92971B92.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/4CA92E6E-D5FC-DF11-88F3-0018F3D09684.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/3ACA31E8-D5FC-DF11-AB02-001A928116C2.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/2C30DE6B-D5FC-DF11-AB12-00248C55CC40.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/121341DD-D3FC-DF11-9234-00261894386E.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/0AB3A269-D5FC-DF11-A284-001A92810AA2.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START310_V1_FastSim-v1/0100/0457E970-D5FC-DF11-A1EF-001A92810AA2.root'


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
