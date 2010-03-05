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
photonValidation.OutputFileName = 'PhotonValidationRelVal350_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(


# official RelVal 350 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V21-v1/0013/5CA21259-4C13-DF11-8368-0030486790C0.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V21-v1/0013/2C2C67CE-4B13-DF11-9D50-002618943959.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V21-v1/0013/2AE64648-4913-DF11-92C6-001A92971B90.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V21-v1/0013/2843A9B1-4A13-DF11-919A-0018F3D095EC.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V21-v1/0013/1C96194C-6213-DF11-BF01-0018F3D09690.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V21-v1/0013/0055F152-4A13-DF11-BB08-001BFCDBD1BA.root'      
    ),
    secondaryFileNames = cms.untracked.vstring(


# official RelVal 350 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/F621264C-4A13-DF11-9DC4-0018F3D0965E.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/EE602750-4A13-DF11-8A05-001731AF67E3.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/EC69CD4B-4C13-DF11-A06C-002618FDA263.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/EA67360E-4813-DF11-93DE-001731AF6A4F.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/E61948C8-4B13-DF11-90B0-0026189438C9.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/E2A99447-4913-DF11-B3DC-003048679006.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/E263C46A-4B13-DF11-A9BA-0018F3D095FC.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/C8604F5B-4C13-DF11-BEDF-002618FDA207.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/ACEB0746-4913-DF11-8BB5-003048678F06.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/8E6AA457-6213-DF11-94BD-0018F3D09600.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/7E58B952-4C13-DF11-B331-001731AF68B7.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/6C742651-4A13-DF11-AE7D-001BFCDBD1BA.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/0878229D-4913-DF11-AA87-003048679030.root',
        '/store/relval/CMSSW_3_5_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/00982D64-4B13-DF11-AD3C-003048679296.root'
    
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
