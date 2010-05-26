
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
process.GlobalTag.globaltag = 'MC_3XY_V15::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal350pre3_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(


# official RelVal 350pre3 QCD_Pt_80_120
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/B0026249-D103-DF11-A139-0030487A195C.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/9CDC89D9-0803-DF11-9D10-0030487CD13A.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/9C979251-0403-DF11-9C5F-0030487A17B8.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/9022DC83-0403-DF11-B5BA-001D09F23A20.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/8ACA874E-0503-DF11-B21A-0030487A1FEC.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/3CB6012F-0303-DF11-8183-0030487C60AE.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/1E6C658F-0B03-DF11-8913-0030487D1BCC.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/1A01CF8D-0303-DF11-9F31-0030487CD77E.root'
        
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 350pre3 QCD_Pt_80_120

        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/F8F345EF-0403-DF11-A817-0030487C5CFA.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/E230EC55-0303-DF11-B0AB-0030487A17B8.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/D47143D0-0803-DF11-9C1D-0030487A3DE0.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/BA0A80F1-0303-DF11-95A4-0030487C8E02.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/98C6B553-D103-DF11-9E5B-0030487A17B8.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/8E52EF6F-0903-DF11-8AD5-0030487A3DE0.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/8AF22056-0503-DF11-BCAF-0030487C8E02.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/78270AFC-0203-DF11-9A4E-0030487C6A66.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/762DE64C-0403-DF11-B2D5-0030487C635A.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/66B6503F-0603-DF11-ABC7-0030487A3C9A.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/5CD9532B-0303-DF11-8C5D-0030487A3C92.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/58DEEC7B-0403-DF11-A5DF-0030487A18A4.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/4EF44D88-0303-DF11-8E26-0030487A1884.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/44EB424B-0403-DF11-B8D7-0030487C8E02.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/388A4086-0B03-DF11-AC1A-0030487C6090.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/0E2A2AE7-0403-DF11-AFD7-0030487A3232.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/0C17772E-0303-DF11-9BCE-0030487CD13A.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v1/0005/06B28BF7-0303-DF11-A493-0030487C6A66.root'

        
     
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


