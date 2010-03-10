
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
process.GlobalTag.globaltag = 'MC_3XY_V24::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre2_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre2 QCD_Pt_80_120

        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0002/D6F0EB0F-A827-DF11-9BDB-003048678AC0.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0001/9EBD5B5A-4A27-DF11-AB8B-00261894392D.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0001/06D50633-6F27-DF11-8699-002618943926.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0000/AE335F4E-FA26-DF11-8704-001A92971B78.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0000/7C0D91D3-FA26-DF11-8C7A-003048678FEA.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0000/6C7631E0-F926-DF11-857D-001731AF6943.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0000/649BCBB7-F726-DF11-8C59-0026189438ED.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0000/220F0B54-F926-DF11-8FFF-002354EF3BE3.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V24-v1/0000/18ED8E52-F826-DF11-9990-001A92971B78.root'
        
        
        ),
    
    
    secondaryFileNames = cms.untracked.vstring(
        
        # official RelVal 360pre2 QCD_Pt_80_120
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/70248017-A827-DF11-96E2-003048D25B68.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/F04FC12C-6F27-DF11-A195-00248C55CC62.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/283B5CE0-4827-DF11-B7FA-00261894383C.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/02FB8BDD-4A27-DF11-B8E5-0026189438FC.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/E4C4944C-FA26-DF11-ACC8-003048679164.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/E02626DF-FA26-DF11-887B-0030486790B8.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/DE52DC55-F926-DF11-89FE-00261894396D.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/BACDB5CF-F926-DF11-9DE4-001A92971B7C.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/A0D4344A-FA26-DF11-A352-003048678F74.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/8EAC8850-F826-DF11-9B41-002618943976.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/82E1C9D5-F926-DF11-BD7D-001731A2897B.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/64ACEF4F-F826-DF11-8E07-002354EF3BDD.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/60DA2AB9-F726-DF11-A558-001A9281170A.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/5C430F51-F826-DF11-B3A1-001A92811746.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/581CF553-F926-DF11-9190-002618943948.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/501B9729-F426-DF11-A577-00261894383A.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/4C7C3BDB-FA26-DF11-9C2E-002618943845.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/4A1EC6D9-F926-DF11-8119-0026189438A7.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/304E5657-F926-DF11-963F-001A928116D8.root'
     
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


