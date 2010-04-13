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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre6_PhotonJets_Pt_10.root'



photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre6 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V4-v1/0011/BAC00DFD-4945-DF11-A2B4-003048679188.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V4-v1/0011/8E1A2BAB-A644-DF11-BE80-00261894383C.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V4-v1/0011/7CB9E1AA-A944-DF11-B46A-00248C55CC7F.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V4-v1/0011/1ED6A622-A444-DF11-A9CA-00261894388A.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V4-v1/0010/90BD3AF5-A244-DF11-885A-00248C55CC9D.root'
 
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360pre6 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/DA924124-A844-DF11-8586-0030486791DC.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/BA47DF81-A444-DF11-B945-002618943842.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/9A75E936-AA44-DF11-9DB6-002618943916.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/7EDA4AA9-A644-DF11-ABE0-00261894395B.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/74CC6398-A744-DF11-A884-001A92810AE6.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0011/62B13F1A-A444-DF11-BA30-002618943836.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/FCC5CEF6-A244-DF11-BA2D-001A9281170E.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/DE64DC6C-A344-DF11-B074-002618943880.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/40017383-A244-DF11-95CE-002618FDA265.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0010/08C562D1-A144-DF11-9E6C-00261894383B.root'
    
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
