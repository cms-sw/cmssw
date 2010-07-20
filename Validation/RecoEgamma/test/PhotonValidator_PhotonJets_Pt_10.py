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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre7_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre7 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V4-v1/0002/FAB44BF9-2B86-DF11-8B9A-0030487CD7EE.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V4-v1/0002/B400F42F-2E86-DF11-912F-0030487C912E.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V4-v1/0002/AA361833-8086-DF11-BD04-003048F11942.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V4-v1/0002/447603C3-3086-DF11-B504-0030487CD16E.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V4-v1/0002/0C0F30F9-2B86-DF11-A87E-0030487A3C92.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V4-v1/0002/00CBFF86-2D86-DF11-A644-0030487CD6D8.root'
 
    ),


    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre7 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/FED99B8F-2B86-DF11-9127-0030487A3C92.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/F0F39F61-8086-DF11-9BB4-000423D9A2AE.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/F0E313EE-2B86-DF11-8D30-0030487C912E.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/C46174F0-2B86-DF11-80B9-0030487C6A66.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/C0942E77-2F86-DF11-98AF-0030487D05B0.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/AA8CD6A2-2E86-DF11-84A6-0030487CD178.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/7A57DDFE-2B86-DF11-8403-0030487CD76A.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/76E5963B-2E86-DF11-A887-0030487CD840.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/70FF1B8F-2D86-DF11-BC83-0030487A195C.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/5EB4EDF8-2B86-DF11-A39C-0030487CD17C.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/0CA3A540-2B86-DF11-855C-0030487C6062.root'

    )
 )


photonPostprocessing.rBin = 48
## For gam Jet and higgs
photonValidation.eMax  = 100
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonPostprocessing.eMax  = 100
photonPostprocessing.etMax = 50




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
