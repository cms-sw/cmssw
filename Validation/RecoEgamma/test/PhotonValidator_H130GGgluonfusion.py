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
process.GlobalTag.globaltag = 'START39_V5::All'

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

photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre5_H130GGgluonfusion.root'

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
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V5-v1/0086/F68DEDCB-18F5-DF11-9444-0018F3D0969C.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V5-v1/0085/A8101DEE-F1F4-DF11-B2A6-001A92810A96.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V5-v1/0085/A0B26008-F0F4-DF11-A47E-001A92971B0C.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V5-v1/0085/A08BFA72-EAF4-DF11-897E-0026189437FA.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V5-v1/0085/08677995-F1F4-DF11-927E-003048678E92.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0088/4687385C-76F5-DF11-B496-002618943862.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/FCDE8396-F2F4-DF11-9AAE-0026189438C2.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/F2F01A0B-EEF4-DF11-A45F-001A92971B16.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/B252BCAC-F2F4-DF11-820C-001A92971B9A.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/B00BBAB8-FAF4-DF11-B6E4-001A928116FC.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/A4FAAA08-F0F4-DF11-AC2D-0030486790BA.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/90ED77FA-E8F4-DF11-A340-002618943985.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/60815473-E9F4-DF11-8CDC-001A928116EA.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/5A603204-E8F4-DF11-B7FB-003048678B44.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/4C9D9825-EBF4-DF11-B477-00261894394A.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/30FFDD8C-E9F4-DF11-BE24-00304867C04E.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/26C27A23-EEF4-DF11-BD8E-003048679030.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/24FA12A8-F0F4-DF11-8A88-00304867BFF2.root'

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
