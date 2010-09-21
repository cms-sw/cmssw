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
process.GlobalTag.globaltag = 'START38_V12::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal384_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 384 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V12-v1/0025/903802FB-9AC2-DF11-B977-003048678FFA.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V12-v1/0024/926CD3F9-80C2-DF11-A294-0018F3D095F0.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V12-v1/0023/F214A415-78C2-DF11-8C1F-00261894388F.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V12-v1/0023/8C1ADBF9-76C2-DF11-AAFB-003048678B44.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V12-v1/0023/5C9DE9A5-78C2-DF11-BA8C-001A928116C2.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V12-v1/0023/323AE381-76C2-DF11-B165-003048678AE4.root'


    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 384 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0025/0E79B0FC-9AC2-DF11-9783-002618943838.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/7CDE877B-80C2-DF11-AD7C-00261894394B.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/4E852778-7FC2-DF11-B298-003048678FDE.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/FC548AF6-75C2-DF11-B9D8-003048678ED2.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/EE8F98A9-78C2-DF11-B545-002618FDA204.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/D0A1BA84-78C2-DF11-8F16-002618FDA211.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/C8360C29-78C2-DF11-ABDD-0026189438DE.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/A2CA0C7E-78C2-DF11-AEB2-0026189438B0.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/5E6F9AFA-76C2-DF11-9B8C-001A928116E6.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/2EC6957E-76C2-DF11-9FE0-0018F3D09688.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/1C418EF9-76C2-DF11-BE99-00304867C0F6.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/12E5DC74-77C2-DF11-9930-002618943973.root',
        '/store/relval/CMSSW_3_8_4/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/08D0EB7B-76C2-DF11-B125-003048678CA2.root'



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
