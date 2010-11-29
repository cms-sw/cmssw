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

photonValidation.OutputFileName = 'PhotonValidationRelVal394_H130GGgluonfusion.root'

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
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V6-v1/0001/B60287AC-16F8-DF11-B378-001A9281173E.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V6-v1/0001/7CB0E1A5-17F8-DF11-890E-002618FDA207.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V6-v1/0001/1CA37789-1FF8-DF11-8182-001A9281174C.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V6-v1/0000/F615C767-FAF7-DF11-BECC-002618943963.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V6-v1/0000/7AD793E7-EAF7-DF11-AA52-002618943838.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
 
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/FE2A5DAF-36F8-DF11-A7E2-002618943863.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/BE197EAA-16F8-DF11-97C9-0026189438B9.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/7C5AEF16-17F8-DF11-B48C-002618943862.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/6E9F8A1D-16F8-DF11-A2CA-001BFCDBD15E.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/6A8AB69F-17F8-DF11-804B-002618943926.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/3C74EB1C-18F8-DF11-A046-0018F3D096E8.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/16546C8B-1EF8-DF11-90A0-002354EF3BD0.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/FC0BBBC7-FAF7-DF11-BF07-0018F3D09600.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/CC972A35-EAF7-DF11-81FD-003048678B04.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/888EFFEE-E4F7-DF11-8D48-001A92971B5E.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/548D817A-F8F7-DF11-8A4B-0018F3D09600.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/12D22FD6-FFF7-DF11-9C1E-001A92971B0E.root',
        '/store/relval/CMSSW_3_9_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/00E23DE3-F0F7-DF11-B02C-002618943975.root'


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

process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
