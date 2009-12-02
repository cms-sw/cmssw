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
process.GlobalTag.globaltag = 'MC_31X_V9::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal335_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 335 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V8D-v1/0008/1EAC9092-F1DB-DE11-86FB-001A92810AB2.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V8D-v1/0007/D0C8E5F9-D9DB-DE11-9EDB-0018F3D096BC.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V8D-v1/0007/B4E60575-D8DB-DE11-B09A-002618943862.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V8D-v1/0007/7A214210-D8DB-DE11-AF12-002354EF3BDB.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP3X_V8D-v1/0007/44320D0A-D9DB-DE11-9B3A-0026189438B5.root'                          
                            
  
 
    ),
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 335 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0008/4C55C132-12DC-DE11-A36B-001A928116E0.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0007/F6B5520C-D9DB-DE11-BAE6-0018F3D09670.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0007/C0615DF4-D9DB-DE11-8175-002618943915.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0007/9C3C0277-D8DB-DE11-83B1-0026189438DE.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0007/88E4F076-D8DB-DE11-9379-002618943966.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0007/74215314-D8DB-DE11-93B5-0018F3D09682.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0007/6C136E0F-D8DB-DE11-A2D9-002618943956.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0007/62C49F13-D8DB-DE11-A981-001A92810AEC.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0007/60A89AF8-D9DB-DE11-A4CC-001A92810A9A.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0007/2A0FB978-D8DB-DE11-9AB5-003048679012.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0007/1CFC3B47-DADB-DE11-8DAE-002618943899.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0007/0AAEE508-D9DB-DE11-B7CF-003048679168.root',
        '/store/relval/CMSSW_3_3_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8D-v1/0007/02F90475-D8DB-DE11-891B-003048678BF4.root'

    
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
