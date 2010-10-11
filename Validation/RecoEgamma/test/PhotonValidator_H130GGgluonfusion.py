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
process.GlobalTag.globaltag = 'START39_V2::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre7_H130GGgluonfusion.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V2-v1/0044/ACF438F7-65D3-DF11-BFBF-00261894390B.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V2-v1/0044/963A7BEB-6DD3-DF11-90C4-001A92811730.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V2-v1/0044/8ECA84DD-67D3-DF11-94B9-00248C0BE014.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V2-v1/0044/8EC5B2E0-68D3-DF11-93D7-003048678B94.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-RECO/START39_V2-v1/0044/0CA1AD66-66D3-DF11-A03A-0018F3D0969C.root'

    ),
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0046/38CCD44B-BCD3-DF11-BA71-002618943838.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/F401F967-67D3-DF11-A1F7-002618943964.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/CA966484-65D3-DF11-AF4D-001A928116BC.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/C4BEA1F7-65D3-DF11-9A97-00261894395F.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/C0B6C5F3-65D3-DF11-8345-002618943964.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/B06AABDB-67D3-DF11-BE82-003048678AE4.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/9ACF156D-68D3-DF11-A1E0-003048D42D92.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/760B6EEF-66D3-DF11-AC9A-002354EF3BE0.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/6E224D64-6DD3-DF11-8996-002618943916.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/68901667-66D3-DF11-82E9-00248C0BE014.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/3AD49ADE-68D3-DF11-AF79-003048679080.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/34CBFDDE-69D3-DF11-A3B6-0026189438A5.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/10E64B85-65D3-DF11-989F-001A92971B0E.root'


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
