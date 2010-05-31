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
photonValidation.OutputFileName = 'PhotonValidationRelVal370_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V4-v1/0026/028B3ACD-8E69-DF11-8530-002618943877.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V4-v1/0024/EA7C9196-3A69-DF11-B02C-00304867BF18.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V4-v1/0024/C450BF07-3969-DF11-8F9A-00248C55CC97.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V4-v1/0024/BE5CCD08-3869-DF11-9158-001A92971B68.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V4-v1/0024/4E27E978-3869-DF11-BEEE-002618943925.root'

    ),


    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370 RelValPhotonJets_Pt_10

        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/F004242F-4069-DF11-8EF2-0026189438AB.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/E8866477-3869-DF11-A9B8-001A92811706.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/9E7CFB6C-3769-DF11-BFF5-00261894389D.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/94E71979-3869-DF11-9738-001A928116E2.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/760B7D95-3A69-DF11-9912-001A928116E8.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/5A11AFFD-3869-DF11-B38F-002618943862.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/5254F1FD-3869-DF11-87DA-002618943882.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/1C4C10F6-3769-DF11-8EEC-001A92971B9A.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/0CF03077-3969-DF11-8AC2-002618943982.root',
        '/store/relval/CMSSW_3_7_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/088A4E74-3869-DF11-B641-001A928116BA.root'


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
