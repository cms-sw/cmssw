
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
process.GlobalTag.globaltag = 'START38_V12::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal384_QCD_Pt_80_120.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 384 QCD_Pt_80_120
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V12-v1/0024/FC8E265A-96C2-DF11-916A-002618943972.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V12-v1/0024/DEC0BB92-87C2-DF11-A6D3-003048679296.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V12-v1/0024/D8BA318A-8AC2-DF11-8D89-001A92971BB8.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V12-v1/0024/A8ED2876-8BC2-DF11-A444-001A92810AEA.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V12-v1/0024/9A3BA991-87C2-DF11-AEFF-00304867D836.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V12-v1/0024/26F3E2FB-86C2-DF11-B449-003048679296.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V12-v1/0024/0A98BFFB-85C2-DF11-9AA0-001A92810A98.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V12-v1/0024/08E001FB-85C2-DF11-BB97-001A92971BA0.root'

     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 384 QCD_Pt_80_120
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/E88659F4-8AC2-DF11-B405-001A92810AEA.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/E0445690-87C2-DF11-9A1B-003048678BAA.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/D83189F6-86C2-DF11-AEBF-00304867BFF2.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/D6BAC8F9-85C2-DF11-B302-001A928116D4.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/D00ADCFA-86C2-DF11-8CC1-0018F3D096DA.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/CA9AE987-8AC2-DF11-8648-0018F3D0965A.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/C87D40FA-85C2-DF11-B10F-0018F3D096DC.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/B848FDF9-86C2-DF11-A7F1-002618943821.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/AC11CF5B-96C2-DF11-9F6A-0026189438F4.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/A6D191F9-85C2-DF11-9C0A-0018F3D0962E.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/A078AFF9-85C2-DF11-942A-0018F3D095EC.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/7A486BF3-8AC2-DF11-A227-0018F3D09686.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/728A367B-85C2-DF11-8DCE-00261894396A.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/6005007A-86C2-DF11-8939-0018F3D096D2.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/52BC3E74-8BC2-DF11-841E-0018F3D09686.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/2C0A3B90-87C2-DF11-BD1E-003048D42DC8.root',
        '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/1A98BDFB-86C2-DF11-BB89-003048679296.root'

        
 
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


