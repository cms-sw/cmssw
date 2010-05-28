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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre5_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre5 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V4-v1/0024/6280AE2C-8F63-DF11-A9D5-00304867906C.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V4-v1/0023/DA3BF6AB-7B63-DF11-A192-00304867BECC.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V4-v1/0023/AA69784F-8563-DF11-A36E-00304867BFAE.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V4-v1/0023/6C51ADC5-7B63-DF11-BD25-00248C0BE005.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V4-v1/0023/289DF950-7E63-DF11-8626-003048679188.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START37_V4-v1/0023/1C45197F-7B63-DF11-9D66-00261894392C.root'

    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre4 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/EE893178-7B63-DF11-8066-002354EF3BDD.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/E62580BF-8363-DF11-B942-003048679076.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/DAF05295-7B63-DF11-B539-00304867BECC.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/D2083FA0-8863-DF11-B79F-00304867BECC.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/B48A3DB0-8463-DF11-95B0-00261894384F.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/A8383FC0-7B63-DF11-B4CF-0026189437F0.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/9C0B5B9C-7B63-DF11-BD03-003048678AC0.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/8811628A-7B63-DF11-8F1E-003048678B86.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/72E517C1-7B63-DF11-9FEC-0026189438B3.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/62B0E541-7C63-DF11-BC75-003048679188.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/5219559D-7B63-DF11-9FA5-003048678B1A.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/30E48AC3-7B63-DF11-91AF-002618943800.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/22182B3F-8563-DF11-B316-00261894384F.root'


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
