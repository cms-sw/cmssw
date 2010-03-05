
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
process.GlobalTag.globaltag = 'MC_3XY_V21::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal352_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

# official RelVal 352 QCD_Pt_80_120
   '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0016/ECAF1151-D91E-DF11-ABE2-003048678B06.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0015/F24E0EC9-241E-DF11-95D4-003048678F0C.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0015/EA9BE544-281E-DF11-A2C6-001A92971ADC.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0015/E8C24FB6-221E-DF11-B348-001A92810AD6.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0015/C47F3608-291E-DF11-B26A-003048678BAA.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0015/8AA86C79-241E-DF11-A4E4-00304867C0C4.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0015/523DC315-201E-DF11-B663-0026189438BF.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0015/4C2972E4-251E-DF11-AB0D-0018F3D09680.root'
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 350 QCD_Pt_80_120

        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/44BF7C4C-D91E-DF11-BE98-003048678B12.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/FE41C7E7-251E-DF11-8F75-002618943894.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/F61DF018-201E-DF11-BE8A-002618943982.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/F44439A2-281E-DF11-AACE-003048678B08.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/EE202D22-201E-DF11-9D20-003048678B1C.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/E2457E32-231E-DF11-8E7D-0018F3D096CA.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/CC095E1B-201E-DF11-8845-00261894397D.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/C889BFE4-231E-DF11-B3F3-0018F3D096F6.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/B80478DF-231E-DF11-9E52-0030486792A8.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/B6775DE6-251E-DF11-8034-002618FDA277.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/B609B64C-251E-DF11-B2A9-003048678B8E.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/984E6980-261E-DF11-8FD3-001731AF6721.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/86A4F809-291E-DF11-BBCA-001731A284C5.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/742D09C8-241E-DF11-B889-00261894387D.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/6CBCA2E1-261E-DF11-A525-00304867918A.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/54C1B7C8-241E-DF11-9E42-00261894384A.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/14984D19-201E-DF11-8A7B-0026189438F3.root',
        '/store/relval/CMSSW_3_5_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0015/001B3D15-2A1E-DF11-BD75-003048678A88.root'
     
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


