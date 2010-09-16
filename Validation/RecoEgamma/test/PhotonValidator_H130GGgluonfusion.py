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
process.GlobalTag.globaltag = 'START38_V9::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal383_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 383 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0022/C6D6F602-3BC0-DF11-A45B-003048678FB8.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0022/C23C5899-E1BF-DF11-B87A-001A92971B7E.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0022/94A499E8-EEBF-DF11-B94E-0018F3D09676.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0022/84CA81E4-EFBF-DF11-B3C2-00261894396E.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0022/00B72CD4-F6BF-DF11-A1A2-003048678FA6.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V9-v1/0021/BE2A2363-95BF-DF11-97C2-003048678B30.root'

    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 383 RelValH130GGgluonfusion
'/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/E6E4BBE8-EEBF-DF11-B0B0-0018F3D096F0.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/E6B062E3-EFBF-DF11-B345-002618943981.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/BED846E7-EEBF-DF11-828A-002354EF3BDA.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/BC0738E7-EEBF-DF11-8813-0018F3D09676.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/9ED93791-E1BF-DF11-BA07-0018F3D09678.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/98C558EA-EFBF-DF11-989A-0026189438FA.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/32F6E3E4-EFBF-DF11-B3E4-0018F3D096DC.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/1A4E41D6-F7BF-DF11-B0F5-0018F3D09652.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/129A9AF5-3AC0-DF11-BA4B-001A928116D8.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0022/0E4F6961-F0BF-DF11-96E4-002618943843.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/CE71C858-94BF-DF11-A539-00261894398B.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/C02C58DE-95BF-DF11-8E60-00261894382D.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/9CFF8456-92BF-DF11-81B0-002618943921.root',
        '/store/relval/CMSSW_3_8_3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0021/68B7FBDD-9CBF-DF11-A571-001A928116EA.root'

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
