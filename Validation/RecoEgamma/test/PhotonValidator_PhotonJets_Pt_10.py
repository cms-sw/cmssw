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
process.GlobalTag.globaltag = 'START38_V12::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal384_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 384 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V12-v1/0024/4E9C600C-96C2-DF11-B95B-003048D3FC94.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V12-v1/0023/FC5B307C-70C2-DF11-8847-003048678BC6.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V12-v1/0023/D0AC5D3F-6FC2-DF11-B902-001A928116AE.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V12-v1/0023/AED875F4-6FC2-DF11-9914-001A92811728.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START38_V12-v1/0023/52C19417-6FC2-DF11-8F27-003048678DA2.root'


    ),


    secondaryFileNames = cms.untracked.vstring(
# official RelVal 384 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0024/98315D0D-96C2-DF11-9888-001A92971BB8.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/CC9D4DF3-6FC2-DF11-8994-0026189438F6.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/BE419A02-6FC2-DF11-9B38-00304867BFA8.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/9081D57C-70C2-DF11-9AE3-003048678C06.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/74E68C37-6FC2-DF11-8720-003048678B00.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/62890EF4-6DC2-DF11-8374-0018F3D0967A.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/6032A8F2-6FC2-DF11-9510-0018F3D096C8.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/3461A773-6FC2-DF11-96A9-0026189438BA.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/2865BD3D-6FC2-DF11-82DF-0018F3D09620.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/20E03BED-73C2-DF11-AABA-0026189438F2.root',
        '/store/relval/CMSSW_3_8_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V12-v1/0023/2097D037-6FC2-DF11-9055-0026189438BA.root'
 

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
