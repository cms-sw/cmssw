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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre2_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

# official RelVal 360pre2 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0001/4E539BE7-6E27-DF11-B4FE-002618943926.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0000/C439FDFA-0127-DF11-95F4-002618943947.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0000/52057979-0327-DF11-8598-001A92971B7C.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0000/34222D6A-0027-DF11-A3E4-0018F3D0968C.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0000/085A9679-F926-DF11-8148-001731A28319.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V24-v1/0000/04E3B2F5-0327-DF11-9EA6-0026189438D8.root'


    ),
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 360pre2 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/F0FFBB63-0127-DF11-A618-002618943907.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/80C9B089-0527-DF11-A803-002618943875.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/76C49DCB-F926-DF11-A6B6-001731AF6847.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/683477E1-0227-DF11-BF43-003048678B76.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/5E0FF547-F826-DF11-A98A-001A92810A96.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/5CED9FFD-0127-DF11-87F0-0017312B554B.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/56DE1A7D-0327-DF11-B018-0017313F01E4.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/4E4AD46F-0227-DF11-880C-001731AF6A4D.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/4C5DD778-0327-DF11-89AE-0017312B56A7.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/32B382EE-0027-DF11-84FE-002618FDA28E.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/1ACCD966-F926-DF11-9307-002618943939.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/1A63A27A-F926-DF11-89AC-001731A2897B.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/10C6A8FC-0327-DF11-B5E5-0017313F01E4.root'

    
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
