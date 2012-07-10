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

photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre7_H130GGgluonfusion.root'

photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName
photonPostprocessing.OuputFileName = photonValidation.OutputFileName

conversionPostprocessing.standalone = cms.bool(True)
conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName
conversionPostprocessing.OuputFileName = tkConversionValidation.OutputFileName


process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START310_V1-v1/0102/061ECB7A-39FD-DF11-ADD0-001A9281171C.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START310_V1-v1/0101/F2A49D95-F7FC-DF11-AFFC-003048678FA0.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START310_V1-v1/0101/94BB7602-F9FC-DF11-BD4E-0026189438BF.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START310_V1-v1/0101/7C47D36B-F6FC-DF11-84BA-002618FDA28E.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START310_V1-v1/0101/7094ECFB-F7FC-DF11-9C46-002618943900.root'


    ),
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0103/96809F52-45FD-DF11-860F-002618943919.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/F4B52596-F7FC-DF11-B6A1-00261894397F.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/D0A7BEA0-F9FC-DF11-BA56-0018F3D0967A.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/8E1C5E03-F7FC-DF11-A84A-001A92810AB8.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/78A724FF-F6FC-DF11-B824-002618943974.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/78060A97-F7FC-DF11-AB40-001A928116DA.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/6C184803-F7FC-DF11-BD2B-0018F3D09684.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/563CC900-F7FC-DF11-ABF0-003048678FA0.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/54C446A0-F9FC-DF11-A94C-0018F3D0965E.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/3AEB3E06-F8FC-DF11-BB2D-003048678B8E.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/3AAE42F4-F7FC-DF11-B9BB-0018F3D096A2.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/1CBDE394-F7FC-DF11-813C-0026189438DE.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/1063BD6D-F6FC-DF11-BC45-0018F3D09702.root'

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

process.p1 = cms.Path(process.tpSelection*process.photonPrevalidationSequence*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
