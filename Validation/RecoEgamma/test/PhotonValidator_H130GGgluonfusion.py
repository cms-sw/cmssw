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
process.GlobalTag.globaltag = 'START3X_V25::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal355_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 354 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V25-v1/0006/F6C22931-BF37-DF11-BC7D-00304867D836.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V25-v1/0006/B615A4B4-BD37-DF11-B4AF-00261894388F.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V25-v1/0006/A68BA4E8-C837-DF11-B670-003048678BE8.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V25-v1/0006/267B7F57-C437-DF11-A428-003048678FEA.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V25-v1/0006/1EAB992D-C037-DF11-8AC6-002618943967.root'

    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 354 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0007/1A0F9BCC-0F38-DF11-885F-003048678BAA.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/E8106C2B-C037-DF11-AC6E-001A92971B5E.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/D62580A6-BF37-DF11-A666-002618943974.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/CA288357-C437-DF11-81DE-003048678F8C.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/B89187A5-C037-DF11-856C-001A928116E0.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/80C59903-BE37-DF11-8F16-00261894395B.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/70929EA4-BF37-DF11-907D-002618943907.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/5AE96BB1-BD37-DF11-9439-002618943867.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/54E6331E-BF37-DF11-99E4-002618943876.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/4C369BD9-C737-DF11-915F-002618943971.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/466A45CC-C537-DF11-8531-002618943932.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/32C3D21E-BF37-DF11-86B2-00261894395C.root',
        '/store/relval/CMSSW_3_5_5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/2A8097FC-BC37-DF11-A225-002618943868.root'
    
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
