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

photonValidation.OutputFileName = 'PhotonValidationRelVal392_H130GGgluonfusion.root'

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

        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V3-v1/0070/C6587EFC-37E8-DF11-AFD7-001A92971B3A.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V3-v1/0070/AC51ECFA-37E8-DF11-9CD5-001A92810AEA.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V3-v1/0070/6ABBEFF7-37E8-DF11-9004-001A928116BA.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V3-v1/0070/38BAC9FB-37E8-DF11-8552-0018F3D0969A.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V3-v1/0070/1669EAF8-37E8-DF11-B3DD-001A92810AC6.root'

 
    ),
    secondaryFileNames = cms.untracked.vstring(
 
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0073/38752221-A8E9-DF11-9D8A-0026189438EF.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0071/F430DC8A-40E8-DF11-BD9D-00304867C0EA.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0071/F20A1089-40E8-DF11-955F-002618FDA248.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0071/F0A71F8E-40E8-DF11-A89E-001A92810AE2.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0071/B89DCF89-40E8-DF11-AEB6-001A92811746.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0071/B045D38B-40E8-DF11-B3BF-0018F3D09696.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0071/AA302186-40E8-DF11-8727-003048679214.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0071/88C58A8B-40E8-DF11-92CC-001A928116EE.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0071/7C6EB686-40E8-DF11-8BFA-00304867C034.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0071/6C56F489-40E8-DF11-A0A4-0026189437EB.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0071/56E4EA8E-40E8-DF11-856E-003048D15CC0.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0071/10D0B58C-40E8-DF11-A8E7-001A92811702.root',
        '/store/relval/CMSSW_3_9_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0071/0CCBB08B-40E8-DF11-9A3A-001A92971B72.root'



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
