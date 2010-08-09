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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre1_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre1 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V8-v1/0009/1C445CCA-069B-DF11-B482-003048678FFA.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V8-v1/0008/CCB836E8-CB9A-DF11-8867-00304867920A.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V8-v1/0008/AE636DA5-CD9A-DF11-A569-003048678AE4.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V8-v1/0008/10A5070D-CE9A-DF11-885D-00248C55CC3C.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V8-v1/0008/0E95EA6F-CC9A-DF11-8372-001A928116CC.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V8-v1/0008/0CD017D9-D49A-DF11-9635-00261894394A.root'

 
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre1 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0009/7673CA96-069B-DF11-86E6-003048678FF6.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/F81E395F-D49A-DF11-8CD3-002618943964.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/D48808EE-CB9A-DF11-9F9F-001A9281174A.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/D0EECB94-CD9A-DF11-B713-00304867BFA8.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/BC728369-CC9A-DF11-8693-0030486792BA.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/A43CC169-CD9A-DF11-9D6F-003048678AE4.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/A040F06E-CB9A-DF11-AA97-0018F3D0967E.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/9C2C536A-CB9A-DF11-8C38-0026189438EA.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/4C1344A9-CD9A-DF11-808E-00304867C0C4.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/1E016F6D-CE9A-DF11-BB49-002618943974.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/18B7CB0B-CE9A-DF11-8E2C-00261894386E.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/14F59BEF-CB9A-DF11-8761-0018F3D0968C.root',
        '/store/relval/CMSSW_3_9_0_pre1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V8-v1/0008/10423D69-CC9A-DF11-A60C-003048678AF4.root'

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
