import FWCore.ParameterSet.Config as cms

process = cms.Process("TestConversionValidator")
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.conversionPostprocessing_cfi")
process.load("Validation.RecoEgamma.tkConvValidator_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START39_V3::All'

process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.tkConvValidator_cfi import *
from Validation.RecoEgamma.conversionPostprocessing_cfi import *

tkConversionValidation.OutputFileName = 'ConversionValidationRelVal3_10_0_pre2_QCD_Pt_80_120.root'
#tkConversionValidation.OutputFileName = 'ConversionValidationRelVal3_10_0_pre2_QCD_Pt_80_120_TESTExplicitMergedHP.root'
tkConversionValidation.mergedTracks = True

conversionPostprocessing.standalone = cms.bool(True)
conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName


process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0058/D6F1C1D6-ADE2-DF11-A2DE-003048678B88.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0058/C6E50F8D-A7E2-DF11-9344-001A92971B92.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0058/AEC9EE30-A9E2-DF11-BF6C-001A92971BD6.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0058/725078B3-A7E2-DF11-AB25-003048D15D22.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0058/6C23BF25-A9E2-DF11-A225-003048D3C010.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0058/4AF2DC88-DFE2-DF11-B3CC-001A928116BE.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0058/2AA06C70-A8E2-DF11-98D6-0026189438EF.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0058/1A50B06F-A7E2-DF11-B042-002618943829.root'

     ),
    
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0060/F0A8091C-F0E2-DF11-AE67-003048678F8C.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/F8363F8A-A7E2-DF11-BE38-002618943986.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/E078D8AC-A8E2-DF11-86B7-0026189438C2.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/DCD6BE8A-A8E2-DF11-A838-002354EF3BD0.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/DABCEFA7-ADE2-DF11-936C-0030486791BA.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/C257F101-A7E2-DF11-9FD9-002618FDA277.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/BEF1D3E7-A7E2-DF11-85A0-0026189438B1.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/A69B1E6B-A7E2-DF11-B123-0026189438C0.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/94AEF574-A7E2-DF11-98CC-0026189438FC.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/9494AF70-A7E2-DF11-9CDD-00261894388F.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/625BCC05-A7E2-DF11-A7C6-002618943978.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/58543E81-A7E2-DF11-ACF9-002618943868.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/3AB40148-A9E2-DF11-A22E-0026189438CF.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/2C5BE6C7-A6E2-DF11-8A9B-0026189438D4.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/2C57E796-A8E2-DF11-8A30-002618943962.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/08B118D1-A7E2-DF11-A1C4-003048678A80.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/021D0D72-A7E2-DF11-8E94-00261894380A.root'




        )
 )







process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)



process.p1 = cms.Path(process.tpSelecForEfficiency*process.tpSelecForFakeRate*process.tkConversionValidation*conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
