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
process.GlobalTag.globaltag = 'START36_V4::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal360_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V4-v1/0014/6CE4F991-BC49-DF11-91BF-0026189437E8.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V4-v1/0013/C80E960B-9949-DF11-BC14-0026189437E8.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V4-v1/0013/A0B0E2A1-9649-DF11-8F75-003048678E94.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V4-v1/0013/2621AF86-9949-DF11-A8E9-003048678F78.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V4-v1/0013/14364906-9B49-DF11-B54F-0018F3D095EC.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0014/EC7E8C7C-FC49-DF11-AA33-00304867C0C4.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/FE3237E1-8B49-DF11-BBB7-001A92811702.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/E2F5A169-9749-DF11-910A-00304867C026.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/981DBA1F-8C49-DF11-8FE3-001A928116DE.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/922D6E35-9549-DF11-B655-001A92810AEA.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/90D54C53-8949-DF11-8D16-0018F3D096C0.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/88A84EC0-8B49-DF11-AE68-001A928116DE.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/6880B0EC-9549-DF11-8433-003048678FB8.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/4E567356-8A49-DF11-9957-00304867BFA8.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/403ACBD2-8F49-DF11-86B7-003048679008.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/3CAAEE52-8A49-DF11-AC01-003048678A78.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/2AD0B21E-8C49-DF11-B2C6-001A92811708.root',
        '/store/relval/CMSSW_3_6_0/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/0C42BF55-9A49-DF11-8D46-001A92810AD2.root'

    
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
