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
process.GlobalTag.globaltag = 'START38_V4::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre7_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre7 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V4-v1/0002/40E2AA22-2B86-DF11-B9D5-0030487A3DE0.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V4-v1/0001/FE2476A5-D285-DF11-943F-0030487CD718.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V4-v1/0001/CEEFCF0F-D885-DF11-A929-0030487CD6D8.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V4-v1/0001/A20C2EAE-D685-DF11-920B-003048F118AA.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V4-v1/0001/92D4FEDF-D585-DF11-ADA9-003048CFB40C.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V4-v1/0001/389CABEE-D485-DF11-BCAE-0030487CD77E.root'
        

    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre7 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/FC066537-D685-DF11-AA02-003048F1BF68.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/D8133637-D685-DF11-9092-003048F1BF66.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/CE4400AE-D785-DF11-8DDC-0030487A3C9A.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/A0E16EE5-D185-DF11-85C4-0030487CD7C0.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/9CB95BD5-D085-DF11-9BC1-0030487CD180.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/92354D11-D785-DF11-B788-003048F1182E.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/8E54A6DF-D585-DF11-85C6-003048F1BF66.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/5AF9EFBA-D685-DF11-A89D-003048F118C6.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/46F8F023-2286-DF11-8917-00304879FA4C.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/44677188-D585-DF11-92AF-003048F11C28.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/1A0D0159-D185-DF11-813A-0030487C6A66.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/14DC1301-D585-DF11-A1B8-003048F1C836.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0001/10BC92E3-D385-DF11-8F4B-0030487C8E02.root'

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
