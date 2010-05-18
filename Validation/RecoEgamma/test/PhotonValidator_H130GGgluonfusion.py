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
process.GlobalTag.globaltag = 'START37_V3::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre4_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre4 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V3-v1/0022/6A840133-A85D-DF11-9C92-0026189437F2.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V3-v1/0021/F427D967-735D-DF11-BA1E-0030486792BA.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V3-v1/0021/E0B75B3F-755D-DF11-8445-0026189438D4.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V3-v1/0021/88FEE624-665D-DF11-9E21-0030486792B8.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V3-v1/0021/840C8C4D-665D-DF11-BE38-0026189438BC.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V3-v1/0021/24AEC47E-655D-DF11-972A-003048678C9A.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre4 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0022/20BBCD83-A85D-DF11-AA1C-002618943863.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/F8A8B93C-755D-DF11-B62C-00261894387C.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/F4626435-6C5D-DF11-9C11-0026189437FE.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/AC92CF7F-655D-DF11-82B1-001A928116D2.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/A26D6A28-655D-DF11-8F90-002618943982.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/8EE60F28-655D-DF11-BDD9-002618FDA21D.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/6A9F1C1D-745D-DF11-AFD0-001BFCDBD1BA.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/6208591F-665D-DF11-BE1D-002618FDA237.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/603E5A69-735D-DF11-A3E6-0018F3D0969A.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/44A402A8-655D-DF11-88EB-003048678F74.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/34E85D54-665D-DF11-B88E-002618943902.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/24BACC1E-665D-DF11-81A5-001A92971B84.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V3-v1/0021/20564C0A-725D-DF11-8969-003048679162.root'
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
