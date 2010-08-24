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
process.GlobalTag.globaltag = 'START38_V8::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal381_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 381 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V8-v1/0011/CCEF3BA8-27A2-DF11-9F08-001A92810A9A.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V8-v1/0010/E6F1090A-CDA1-DF11-AD7E-001A92971B8C.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V8-v1/0010/E434D18F-CBA1-DF11-887E-00304867902E.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V8-v1/0010/BA09A516-CBA1-DF11-9BED-001A92971B8A.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V8-v1/0010/984A35FD-CBA1-DF11-AF78-003048D15DDA.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V8-v1/0010/469ABEFD-CBA1-DF11-A9A5-003048678E94.root' 
    ),
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 381 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0011/9226759E-2DA2-DF11-B1AE-0026189438FF.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/F8EEEA01-CCA1-DF11-BF3D-001A92971AEC.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/BEC15701-CCA1-DF11-AC5A-0018F3D09710.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/BCA64108-CBA1-DF11-8FA8-002618943894.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/7C9F7B04-CDA1-DF11-9E99-0018F3D096BE.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/6C689709-CBA1-DF11-8880-00261894388A.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/5C0C070B-CBA1-DF11-8BF9-002618943932.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/523C9181-CCA1-DF11-82CD-001A92810ACA.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/4AE1E4FF-CBA1-DF11-88A2-003048678ED2.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/44BAA2FB-CBA1-DF11-A236-003048678B44.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/26ED658D-CBA1-DF11-A441-002618943860.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/1877E40B-CBA1-DF11-A81F-0026189437F8.root',
        '/store/relval/CMSSW_3_8_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0010/101070FD-CBA1-DF11-8A8A-00304867902E.root'

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
