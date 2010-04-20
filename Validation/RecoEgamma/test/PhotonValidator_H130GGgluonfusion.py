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
process.GlobalTag.globaltag = 'START3X_V26::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal357_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 357 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V26-v1/0012/F4269A74-4749-DF11-A35B-003048678B5E.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V26-v1/0012/B20201F3-4549-DF11-AB1D-00304867C1BC.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V26-v1/0012/5220F773-4849-DF11-BF45-003048678B8E.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V26-v1/0012/0AB79375-4749-DF11-8C4F-002354EF3BE2.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V26-v1/0012/06ABAA29-6949-DF11-9A1E-00304867C0EA.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V26-v1/0012/0446567B-4649-DF11-898C-0030486792B8.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 357 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0013/B653C31C-6949-DF11-8018-002354EF3BDA.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/EE435402-4849-DF11-A9E1-003048679296.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/DE3BEE02-4849-DF11-834E-003048D3FC94.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/D6733F73-4749-DF11-8738-003048678FA0.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/C2E8F772-4749-DF11-A863-0030486792B4.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/B0A810C8-4449-DF11-91AD-003048B95B30.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/9EC78D7D-4649-DF11-ACF6-003048B95B30.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/54B0D072-4749-DF11-9D81-003048679076.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/16B63F52-4549-DF11-82F8-003048679166.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/129216EE-4549-DF11-935F-0030486792B8.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/0A5E597D-4649-DF11-BC29-003048678F62.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/00C61C05-4849-DF11-8E77-0030486790B8.root',
        '/store/relval/CMSSW_3_5_7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/00A8E073-4749-DF11-AEFB-002354EF3BE2.root'
    
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
