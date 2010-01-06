
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
process.GlobalTag.globaltag = 'MC_3XY_V14::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal350pre2_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(


# official RelVal 350pre2 QCD_Pt_80_120

        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0010/F4BBBD32-22EE-DE11-8165-0026189438C1.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/DC21642E-7EED-DE11-AF12-002618943970.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/BC082D2F-7FED-DE11-9EF0-002618943924.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/A4FD8A19-7DED-DE11-9A14-00261894397D.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/7A2ACFA4-7DED-DE11-9EE0-002618943831.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/541D301D-7DED-DE11-AA8E-002618943829.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/464E1A2D-7EED-DE11-B0A4-002618943982.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0009/182C15AB-7DED-DE11-B06A-00248C0BE016.root'


    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 350pre2 QCD_Pt_80_120

        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0010/D63CAA33-22EE-DE11-B886-00261894392C.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/EE5D89A7-7DED-DE11-B008-002618FDA248.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/EA8622A7-7EED-DE11-839B-0026189438C1.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/E05BC727-7FED-DE11-AAB8-002618943829.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/D879FB1B-7DED-DE11-91E9-0026189438D5.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/D8367A1C-7DED-DE11-AAC1-002618943962.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/C0F8BC29-7EED-DE11-A85E-002618943896.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/BE965C2A-7EED-DE11-AD0A-002618943977.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/8A00509D-7DED-DE11-80D4-0026189438F8.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/824E042A-7EED-DE11-813E-002618943970.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/823B1C1C-7DED-DE11-AB5B-0026189438CC.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/42C3421C-7DED-DE11-A35B-003048D3FC94.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/42A99028-7EED-DE11-A99A-0026189438A2.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/4290D128-7EED-DE11-AC2F-002618943821.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/100D251D-7DED-DE11-8B48-00261894388D.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/0C88B32A-7EED-DE11-B426-002618943922.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/04BE5929-7EED-DE11-B670-002618943800.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V14-v1/0009/02970029-7EED-DE11-B097-002618943821.root'

     
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


