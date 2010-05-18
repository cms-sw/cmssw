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
process.GlobalTag.globaltag = 'START36_V7::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal361_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 361 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V7-v1/0020/D00B31DA-2A5D-DF11-9A30-0018F3D09620.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V7-v1/0020/B0675FC7-285D-DF11-949D-0026189438FF.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V7-v1/0020/7ED84C8F-2A5D-DF11-8E39-001BFCDBD1BA.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V7-v1/0020/320B82C3-2A5D-DF11-86D0-001A928116AE.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V7-v1/0020/28D42DB5-325D-DF11-98B1-0018F3D0967A.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 361 RelValH130GGgluonfusion
 
'/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/FE2618C7-2B5D-DF11-B547-0026189438FA.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/FC7E063E-2A5D-DF11-814F-00304866C398.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/FAE7F80C-2A5D-DF11-BCB9-0018F3D096E0.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/F8AFF5D5-2A5D-DF11-B42D-003048678CA2.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/E242C6DB-275D-DF11-9806-0018F3D0970E.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/E0A3A569-295D-DF11-A1F6-003048678FB2.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/C82ABBF3-295D-DF11-9F42-0018F3D096A2.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/B083E109-2A5D-DF11-BDB1-00261894396F.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/9821C5E7-505D-DF11-8E45-001A928116DA.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/66508F53-285D-DF11-A255-001A92971AEC.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/50C7C4D0-285D-DF11-A910-001A928116DC.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/2AD8F694-2E5D-DF11-90AE-002618943861.root',
        '/store/relval/CMSSW_3_6_1/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/2052D707-2A5D-DF11-82A9-002618943865.root'
    
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
