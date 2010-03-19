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
process.GlobalTag.globaltag = 'MC_36Y_V2::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre3_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 354 RelValH130GGgluonfusion


# official RelVal 360pre3 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V2-v1/0005/F2D0B2A9-702F-DF11-A893-001A92971BCA.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V2-v1/0005/E05E37B1-722F-DF11-A04E-001A92971AA8.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V2-v1/0005/B2B60034-702F-DF11-897F-001A92971BB8.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V2-v1/0005/969F87C0-712F-DF11-BEDD-001A92971BB8.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V2-v1/0005/74C40437-712F-DF11-8F66-00261894382D.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-RECO/START36_V2-v1/0005/1CB3AFDD-B02F-DF11-822D-0030486792F0.root'
    ),
    secondaryFileNames = cms.untracked.vstring(


# official RelVal 360pre3 RelValH130GGgluonfusion
  '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/EC66C5BA-712F-DF11-B4B2-00261894382D.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/EAAF0831-712F-DF11-8310-002618943915.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/E0393EA9-702F-DF11-A45A-001A92810AEC.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/DC5B4CAF-702F-DF11-9318-001A92971BD8.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/A24BA32E-712F-DF11-88AC-002618943961.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/A0BBCE31-702F-DF11-950D-0018F3D096C6.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/80A0E631-702F-DF11-AF33-001A92971B16.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/68F008AA-702F-DF11-AE87-0018F3D09648.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/689219AC-702F-DF11-A438-0018F3D09612.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/365C6DBA-712F-DF11-A7E5-001BFCDBD166.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/2EC4C7B9-712F-DF11-AE10-001BFCDBD100.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/0E8876B9-B02F-DF11-BE64-003048678B76.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/D2D50DB6-6F2F-DF11-8756-001BFCDBD11E.root'
  
    
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
