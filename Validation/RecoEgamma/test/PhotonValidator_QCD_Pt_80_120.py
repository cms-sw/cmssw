
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
process.load("Validation.RecoEgamma.tkConvValidator_cfi")
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

photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre5_QCD_Pt_80_120.root'
photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName
photonPostprocessing.OuputFileName = photonValidation.OutputFileName

tkConversionValidation.OutputFileName = 'ConversionValidationRelVal3_10_0_pre5_QCD_Pt_80_120.root'
conversionPostprocessing.standalone = cms.bool(True)
conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName
conversionPostprocessing.OuputFileName = tkConversionValidation.OutputFileName


process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V5-v1/0088/0A6EAA53-76F5-DF11-BF8D-00261894397F.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V5-v1/0085/D6A14A07-FBF4-DF11-AAC6-002618943876.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V5-v1/0085/C81DCF26-F3F4-DF11-8FDB-00304867D446.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V5-v1/0085/9E266265-FCF4-DF11-9E80-001A92971B0E.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V5-v1/0085/9AB7F19B-03F5-DF11-A2FB-0026189438C4.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V5-v1/0085/821B890A-F3F4-DF11-921F-003048678FFE.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V5-v1/0085/3CD38292-F5F4-DF11-8FFA-001A92810AF2.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V5-v1/0085/3075F967-F5F4-DF11-9807-003048679166.root'
     ),
    
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0087/2AF90E33-73F5-DF11-9DEF-002618FDA21D.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/FC33229C-F9F4-DF11-970B-001BFCDBD176.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/F6C89D01-EEF4-DF11-A109-00261894388B.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/F445E4C6-F8F4-DF11-BDEA-002618943896.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/EE78CC58-F7F4-DF11-B7E5-003048678B5E.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/DA03257F-FCF4-DF11-921C-002618943811.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/CEA95920-F1F4-DF11-9480-001A92971BB4.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/B4A21D59-F7F4-DF11-B6AF-0018F3D0970E.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/B2BB7511-07F5-DF11-8D8B-001A92971B64.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/AA4F16E2-EBF4-DF11-B3D9-002618943964.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/94429981-F6F4-DF11-98AD-002618943979.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/84E8CEAD-F4F4-DF11-ABC6-0018F3D0961E.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/6A0B2798-F4F4-DF11-A3B9-002618943957.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/681B0DFC-05F5-DF11-B7DF-002618FDA208.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/3C74C40E-F6F4-DF11-93F4-003048678FD6.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/3AE7B07E-F1F4-DF11-A7B2-0018F3D096AE.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/086C8902-E8F4-DF11-B0EB-00261894391D.root'
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


