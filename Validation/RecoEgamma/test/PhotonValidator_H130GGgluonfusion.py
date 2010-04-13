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
process.GlobalTag.globaltag = 'START36_V3::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre5_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre5 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V3-v1/0009/DCBE3241-D23D-DF11-8F3A-002618943924.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V3-v1/0009/B4C02D43-C43D-DF11-8481-001A92810AB8.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V3-v1/0009/70042285-CA3D-DF11-90AA-0018F3D09630.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V3-v1/0009/34C21518-CC3D-DF11-BBA8-003048678D86.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V3-v1/0009/2EFF2793-CB3D-DF11-BF02-003048679244.root'
 
  
    ),
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 360pre5 RelValH130GGgluonfusion
'/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0010/3870CE7B-483E-DF11-B840-002618943809.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/AC0BFD44-C43D-DF11-B85F-001A92810AD2.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/92B6B094-CB3D-DF11-AB36-002618943857.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/78B1ECB1-C33D-DF11-B29C-0018F3D09608.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/742E90BC-C43D-DF11-AA5C-00304867900C.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/70975443-C43D-DF11-99F8-0018F3D096DC.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/62B2DE9A-CB3D-DF11-B835-0018F3D095EE.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/4A30CB98-CB3D-DF11-86F8-0018F3D09630.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/22C92114-CA3D-DF11-9F6B-001A92811714.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/20A05018-CC3D-DF11-9459-001A92971B12.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/1E3AE1FF-CA3D-DF11-ABED-0026189438B4.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/08B23818-CC3D-DF11-A3D3-001A92971BBE.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/0477E190-CB3D-DF11-883D-002618943986.root'
          
    
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

process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
