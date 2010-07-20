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
process.GlobalTag.globaltag = 'START38_V6::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre8_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre8 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V6-v1/0001/BC19475B-758B-DF11-BA90-00304867C04E.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V6-v1/0001/9EB6758A-7A8B-DF11-B995-0018F3D095FC.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V6-v1/0001/644AA05A-768B-DF11-A97A-0018F3D0967E.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V6-v1/0001/46181018-A88B-DF11-99B9-003048678B14.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V6-v1/0001/127EFB17-7C8B-DF11-9A9B-00248C0BE014.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-RECO/START38_V6-v1/0001/10A060FE-748B-DF11-93D4-0018F3D096C6.root'

 
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre8 RelValH130GGgluonfusion
'/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/F8E3E10A-758B-DF11-8C9D-003048679164.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/F84F6F9E-748B-DF11-BBE0-0018F3D0967E.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/CAA794B3-748B-DF11-9A6A-002354EF3BDE.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/CA72152A-768B-DF11-87C0-001A92971B06.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/BAD606FA-748B-DF11-9EEA-002354EF3BE6.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/AACA9CB2-748B-DF11-A824-0018F3D096CE.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/82FA7CFA-7B8B-DF11-ACC6-002618943836.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/72B54D66-7B8B-DF11-9564-0026189437ED.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/14C1DC14-778B-DF11-B35A-001A92971B06.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/0EB50979-7A8B-DF11-87D3-0018F3D09690.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/0A9F3543-A38B-DF11-A078-0026189438D2.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/08F91200-758B-DF11-BCA8-00304867BECC.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/06FC58B4-748B-DF11-A91A-00304867BEDE.root'
 

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
