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
process.GlobalTag.globaltag = 'START36_V3::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre4_PhotonJets_Pt_10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

# official RelVal 360pre4 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V3-v1/0002/E00983C5-F037-DF11-AA7D-0030487A1FEC.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V3-v1/0002/B6C63380-F237-DF11-A70A-0030487CD16E.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V3-v1/0002/A0F9DEE3-1638-DF11-B841-0030487CD6D2.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V3-v1/0002/A002ED2F-F037-DF11-BE09-000423D60FF6.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V3-v1/0002/4E3441BB-EF37-DF11-B60D-0030487CAF5E.root'
 
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360pre4 RelValPhotonJets_Pt_10

        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0002/D4A01F6F-F237-DF11-AFF8-0030487C635A.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0002/CC69EDB9-F037-DF11-B8B3-0030487A322E.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0002/C4A31A01-F037-DF11-9EC3-0030487A18D8.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0002/AC6DD732-F037-DF11-B57B-0030487C6062.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0002/A44198FB-EF37-DF11-B190-0030487C635A.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0002/9C856AC5-EF37-DF11-8C65-0030487CD77E.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0002/90987108-EF37-DF11-82BC-0030487CAF0E.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0002/7228CDC0-F037-DF11-BB1F-001617C3B654.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0002/28AD5D57-F337-DF11-B118-0030487A1FEC.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V3-v1/0002/0E85D68B-F137-DF11-95B1-0030487C90D4.root'
    
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
