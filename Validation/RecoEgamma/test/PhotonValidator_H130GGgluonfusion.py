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
process.GlobalTag.globaltag = 'START37_V4::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V4-v1/0026/C2B03C86-8E69-DF11-B5C7-001A92971B8A.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V4-v1/0025/DC2E8B11-5469-DF11-A3EE-0030486790FE.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V4-v1/0025/CAA6AF98-5669-DF11-A8DA-001A9281174A.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V4-v1/0025/BE4D92C2-6369-DF11-9F32-0018F3D096C6.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V4-v1/0025/AE4CF904-5369-DF11-96CB-001A92971BDC.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V4-v1/0025/1E9658FC-5169-DF11-817A-00248C55CC7F.root'

    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/EEB468FD-5169-DF11-B1DB-001A92810AC6.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/E4512B4E-8869-DF11-907C-003048678BAC.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/AEA35506-5369-DF11-AF67-0018F3D09676.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/AE76D80E-5469-DF11-8B97-002618943868.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/96EB2899-5369-DF11-A747-001A92810AA0.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/8E441F00-5569-DF11-A43C-003048679164.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/7CF843AF-5769-DF11-8358-003048678B8E.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/72D12E2C-6269-DF11-903A-0018F3D0965A.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/7244A476-5269-DF11-B399-003048678C62.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/7072C497-5369-DF11-862C-00261894387E.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/28CA3870-5169-DF11-9C35-002618943918.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/164ECDC0-6369-DF11-8A3A-0018F3D09686.root',
        '/store/relval/CMSSW_3_7_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/04633EFA-5169-DF11-A0E8-002618943907.root'


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
