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
process.GlobalTag.globaltag = 'START36_V4::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal360_PhotonJets_Pt_10.root'



photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V4-v1/0014/1E798C4B-FC49-DF11-8079-002618943974.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V4-v1/0013/D056E1F9-AA49-DF11-96B9-0026189438DD.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V4-v1/0013/AE800693-9449-DF11-AAFA-001A92971B94.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V4-v1/0013/AC44C9D0-9549-DF11-873E-003048B95B30.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V4-v1/0013/168FB08D-9749-DF11-A58E-0026189438C1.root' 
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0014/FAAFF3E4-A749-DF11-909C-0026189438AB.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/EC86B3FA-9A49-DF11-AAC7-001A92971B74.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/B68F8712-9649-DF11-86AB-001A92810ADE.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/6E7EF185-9449-DF11-AA22-001A9281173A.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/5A3947BE-9149-DF11-8C1D-0030486790B8.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/544B79E6-AD49-DF11-BAA6-00261894387B.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/540D8470-9849-DF11-BC1D-00304867BFB0.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/4E96824F-9749-DF11-A20B-003048678E24.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/36B062E8-9449-DF11-B5DC-00248C55CC97.root',
        '/store/relval/CMSSW_3_6_0/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/22317578-9449-DF11-B1DA-00261894398C.root'   
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
