
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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre5_QCD_Pt_80_120.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre5 QCD_Pt_80_120
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0010/C2112D87-493E-DF11-9EEF-001A9281173C.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0009/F6EE0722-BE3D-DF11-826F-001A928116F0.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0009/D8B85E9C-BE3D-DF11-B6D3-003048678B20.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0009/D0D0521B-BE3D-DF11-9631-003048679084.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0009/9CAE0522-BF3D-DF11-887D-002354EF3BE3.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0009/76DEE3A6-C53D-DF11-B49A-0018F3D0970E.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0009/68DD322E-C23D-DF11-92C4-003048678FB2.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V3-v1/0009/4817C4A0-BD3D-DF11-A3C6-0030486792B8.root'
        
        ),
    
    secondaryFileNames = cms.untracked.vstring(
        # official RelVal 360pre5 QCD_Pt_80_120
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0010/54549090-493E-DF11-89F2-001A9281173C.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/F486AD29-C13D-DF11-A56F-003048678F8E.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/EAAC91A9-BE3D-DF11-BE6C-00261894390C.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/E8EF0B42-C03D-DF11-B191-003048678FFA.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/E896F19E-BD3D-DF11-8ACE-0026189438FD.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/D0446B0A-C53D-DF11-B74C-00261894383B.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/CCC11A16-BE3D-DF11-83B8-003048679228.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/CA4C8712-BD3D-DF11-BE13-00304867C136.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/9C5CEF1B-BF3D-DF11-B9CF-001A928116F0.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/76C80C14-BE3D-DF11-98D6-002354EF3BD2.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/70032918-BE3D-DF11-BDE8-002618943831.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/3E4C149E-BD3D-DF11-83A9-003048679084.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/2664F543-C33D-DF11-A8F4-00304867904E.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/12B5229E-BD3D-DF11-9253-003048678F26.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/1270491B-BE3D-DF11-AD95-003048679164.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/0C10D8A6-BE3D-DF11-AD66-003048678C3A.root',
        '/store/relval/CMSSW_3_6_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0009/02130442-C23D-DF11-9ACB-00304867C29C.root'
        
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


