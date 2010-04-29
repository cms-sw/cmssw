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
process.GlobalTag.globaltag = 'START37_V0::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre1_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre1 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V0-v1/0015/E02FFF59-0E4D-DF11-BEF2-002618943811.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V0-v1/0015/A4C4969B-E64C-DF11-9643-001A928116E8.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V0-v1/0015/7E3A6AC1-E84C-DF11-937C-001A92810ACA.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V0-v1/0015/5EC16407-DD4C-DF11-AF6C-001BFCDBD100.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V0-v1/0015/52C6E584-E04C-DF11-844A-002618943962.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V0-v1/0015/44B2F7AF-DE4C-DF11-88D8-0018F3D09698.root'

    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre1 RelValH130GGgluonfusion
 '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/FC2B6C01-DD4C-DF11-8C24-0026189438BC.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/E0140C1D-E64C-DF11-B073-003048679150.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/DE2B09A9-DE4C-DF11-B8DE-00304867BFB2.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/CA410BAA-DE4C-DF11-9A92-0018F3D096BE.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/9AB5ADF9-DF4C-DF11-9C33-00261894391C.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/6C530C9C-E74C-DF11-B86B-001A92971ACE.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/6AD9999E-E74C-DF11-A64C-0018F3D096D8.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/669F2165-DC4C-DF11-9F38-002618943832.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/56752081-DD4C-DF11-88B5-001A92971BCA.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/4AE8885D-0E4D-DF11-8801-002618943811.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/38B1251E-E64C-DF11-926B-003048678B30.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/1E05D10F-E24C-DF11-8FCF-0026189438BA.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/188EC388-DA4C-DF11-B8F5-00261894387E.root'

    
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
