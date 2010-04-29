
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
process.GlobalTag.globaltag = 'START37_V1::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre2_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre2 QCD_Pt_80_120
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V1-v1/0018/EE2D6A08-F752-DF11-A9F6-00261894397F.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V1-v1/0017/D0E54223-9C52-DF11-8E44-0026189438F2.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V1-v1/0017/B8EDF9C6-9C52-DF11-BF83-0026189438DD.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V1-v1/0017/90EE67AE-9B52-DF11-BC7A-0026189438EA.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V1-v1/0017/54676714-9A52-DF11-92D7-002618943877.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V1-v1/0017/52154643-9F52-DF11-8D5E-003048678B04.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V1-v1/0017/42DE6417-9952-DF11-BEB4-003048678BAE.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V1-v1/0017/407067F8-9D52-DF11-B7BA-003048678B12.root'        
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre2 QCD_Pt_80_120
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0018/FA248C06-F752-DF11-84F9-00261894395C.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/EA03DF42-9E52-DF11-BBA3-003048678B0E.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/CCD4BD10-9952-DF11-90F1-003048678CA2.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/C8BE1F20-9C52-DF11-B230-002618943875.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/C2C9F9C2-9C52-DF11-8A43-002618943978.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/9AA24113-9A52-DF11-9541-003048D15E14.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/76EC359E-9A52-DF11-B71A-002618943969.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/68121BC2-9C52-DF11-B6B1-00261894390B.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/58F04AB1-9B52-DF11-B85E-00304867926C.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/5413D3BF-9F52-DF11-A873-002618943949.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/5064DD42-9D52-DF11-8540-00261894386F.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/3A5031AC-9B52-DF11-A8AE-003048678FE6.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/32CC2A14-9952-DF11-9EE1-0026189437F8.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/305756BB-9C52-DF11-908D-00261894380B.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/28E8F08B-9952-DF11-B400-00304867924A.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/22E314C8-9E52-DF11-8F1F-0026189438B3.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/1E6014F6-9D52-DF11-A7FE-003048678AFA.root'


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


