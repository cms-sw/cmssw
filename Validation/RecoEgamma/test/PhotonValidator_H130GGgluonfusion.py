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
process.GlobalTag.globaltag = 'START37_V2::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre3_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre2 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V2-v1/0019/E69759CD-F457-DF11-AD31-001A92971B16.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V2-v1/0019/CCC377BC-3458-DF11-AF6A-003048678F8E.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V2-v1/0019/B6485BD2-F457-DF11-A75C-001A92971B7E.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V2-v1/0019/625561DC-1258-DF11-A323-001731EF61B4.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V2-v1/0019/525E2B84-F357-DF11-96E6-001A92971BA0.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V2-v1/0019/087E7EBF-F357-DF11-AA96-001A928116B4.root'


    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre3 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/E802CBCB-F457-DF11-AB36-0018F3D0969A.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/B8783EC3-F357-DF11-93D4-0018F3D096E6.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/A4A5F282-F357-DF11-A000-001A92810AD6.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/A2236A00-FB57-DF11-AC38-001A92810A9A.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/986B9AED-F957-DF11-8640-0018F3D096E8.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/7CED04CD-F457-DF11-B65D-0018F3D095FE.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/70E1CCCD-3458-DF11-BD12-003048678F06.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/62802C7E-F357-DF11-A7C4-001A92971B62.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/4A83E2BE-F357-DF11-A0A9-0026189438EF.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/3CD5B8CB-F457-DF11-AF23-001A92810AD8.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/34A6E6CA-F457-DF11-BEFC-0018F3D096E8.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/2E132ECC-F457-DF11-A67A-001A9281172E.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/12C6E94E-F257-DF11-98EB-0018F3D095F2.root'


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
