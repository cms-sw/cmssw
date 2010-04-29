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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre2_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre2 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V1-v1/0018/A6C1088D-F652-DF11-B561-003048678B04.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V1-v1/0017/96F1CB59-AB52-DF11-8DF8-002618943879.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V1-v1/0017/7CE58B47-AB52-DF11-A4B7-002618FDA248.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V1-v1/0017/681B2DB7-AF52-DF11-88E3-00248C55CC97.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V1-v1/0017/347B2525-AA52-DF11-B954-0026189438EF.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V1-v1/0017/3293E718-AC52-DF11-BA88-003048678BAE.root'
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V0-v1/0015/E02FFF59-0E4D-DF11-BEF2-002618943811.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V0-v1/0015/A4C4969B-E64C-DF11-9643-001A928116E8.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V0-v1/0015/7E3A6AC1-E84C-DF11-937C-001A92810ACA.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V0-v1/0015/5EC16407-DD4C-DF11-AF6C-001BFCDBD100.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V0-v1/0015/52C6E584-E04C-DF11-844A-002618943962.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V0-v1/0015/44B2F7AF-DE4C-DF11-88D8-0018F3D09698.root'

    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre2 RelValH130GGgluonfusion
      '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0018/06CD1BAF-F652-DF11-BF9F-0026189438A5.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/F6216C20-AA52-DF11-90D0-00304867905A.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/D4D63523-AA52-DF11-83ED-00304867901A.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/D2BDE711-AC52-DF11-99D6-002618943833.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/D0126796-AA52-DF11-8E1D-002618943879.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/C6B2918B-A952-DF11-90C6-003048D15D22.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/B0316755-AB52-DF11-9DD0-003048678DA2.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/AA9F5B9E-AB52-DF11-BF42-00261894384F.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/94C2029A-AB52-DF11-AF18-0026189438FC.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/921C73B4-AF52-DF11-887C-0026189438B5.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/64E45F46-AB52-DF11-8CBA-002618FDA279.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/3617B518-AC52-DF11-B4B2-003048678BAC.root',
      '/store/relval/CMSSW_3_7_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/0AF6E751-AB52-DF11-9658-00304867926C.root'      
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
