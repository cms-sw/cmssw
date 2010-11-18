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

photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre2_H130GGgluonfusion_2.root'

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


        
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V3-v1/0058/DA7E9CD0-ACE2-DF11-A12E-0030486790A6.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V3-v1/0058/DA59D867-AEE2-DF11-B4CD-003048678C62.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V3-v1/0058/92592207-AEE2-DF11-BF8A-0018F3D0961E.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V3-v1/0058/5A095477-ADE2-DF11-92C5-003048679220.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V3-v1/0058/066F5228-BFE2-DF11-BAC6-00261894389E.root'
 
    ),
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0060/7A7CE41F-F0E2-DF11-9252-002618943886.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/FE3F1031-ADE2-DF11-8EE4-002618943981.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/D64ABD1E-ADE2-DF11-815D-003048678B7C.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/BC84CF58-AAE2-DF11-80BA-00261894392C.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/A6BE38A4-AEE2-DF11-91E7-002354EF3BE6.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/8ED9898B-AAE2-DF11-9AB7-00261894384F.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/880D7D3F-ADE2-DF11-88B5-003048678AF4.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/8288B717-ADE2-DF11-A1A2-003048678B0A.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/725FF96F-ABE2-DF11-B164-0026189438FA.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/6044F9F5-ACE2-DF11-8190-0026189438AE.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/246591E2-ACE2-DF11-AE3E-002618943880.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/2020F3D2-B2E2-DF11-9C5F-00304867915A.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/1E4BFA39-ADE2-DF11-BEC1-00261894394B.root'

 


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
