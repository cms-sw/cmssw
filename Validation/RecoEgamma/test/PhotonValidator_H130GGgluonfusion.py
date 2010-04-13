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
process.GlobalTag.globaltag = 'START36_V4::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre6_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre6 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V4-v1/0011/DED1A110-AF44-DF11-B7EF-001A92810AC6.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V4-v1/0011/66C5897A-AF44-DF11-8B0C-0026189438A0.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V4-v1/0011/5E38B505-B144-DF11-9D66-0018F3D0965A.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V4-v1/0011/16D05B8F-AE44-DF11-AC4C-0018F3D09630.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V4-v1/0011/0CB7B959-A944-DF11-8F16-0026189438C4.root'
    ),
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 360pre6 RelValH130GGgluonfusion
 '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/F0B67A87-AE44-DF11-87A1-002618943843.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/EED7F331-A844-DF11-902B-002618943974.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/EE637404-B044-DF11-A712-0030486791BA.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/BC2B61FF-AF44-DF11-BC9C-00248C0BE013.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/9EFF8DE2-AD44-DF11-B5E3-001A92810AB6.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/922BAF09-AF44-DF11-9703-001A92971ACE.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/8E90F783-AE44-DF11-AFCA-00261894398A.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/8835193B-4A45-DF11-9152-003048679188.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/781BE652-A944-DF11-BF88-002618943959.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/7818C809-AF44-DF11-9DDE-001A9281170C.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/62E3B106-AF44-DF11-802B-0018F3D096EC.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/20636E54-AD44-DF11-ADEF-0018F3D09684.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/1848320C-AF44-DF11-B60B-001A928116D2.root'
    
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
