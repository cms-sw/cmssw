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
process.load("Validation.RecoEgamma.conversionPostprocessing_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START310_V1::All'


process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *
from Validation.RecoEgamma.conversionPostprocessing_cfi import *

photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre7_PhotonJets_Pt_10.root'

photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName
photonPostprocessing.OuputFileName = photonValidation.OutputFileName

conversionPostprocessing.standalone = cms.bool(True)
conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName
conversionPostprocessing.OuputFileName = tkConversionValidation.OutputFileName



process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START310_V1-v1/0101/E2D0FE92-F0FC-DF11-966E-0018F3D09676.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START310_V1-v1/0101/9E2182E7-E8FC-DF11-93DF-003048678B5E.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START310_V1-v1/0101/723E8CC5-ECFC-DF11-9A22-0018F3D095EC.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START310_V1-v1/0101/5279082D-39FD-DF11-8C1B-0018F3D096D8.root'
    ),

    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0103/9A5A0C53-45FD-DF11-BEC6-00261894389A.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/CA6AAA67-E9FC-DF11-AD36-003048D42D92.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/B497BB2C-EDFC-DF11-98AD-00304867BEDE.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/A2C1496B-EFFC-DF11-B999-001A92971ACE.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/8AAC55F9-E6FC-DF11-B92E-00304867C136.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/665A5653-EEFC-DF11-B233-001A92971BBA.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/5E4A8DAA-E1FC-DF11-85DA-003048678FD6.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/56C45505-F2FC-DF11-A699-001A928116D4.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/48284C92-F0FC-DF11-8197-0018F3D09678.root',
        '/store/relval/CMSSW_3_10_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START310_V1-v1/0101/109BB16A-EBFC-DF11-A40C-003048678DA2.root'


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

process.p1 = cms.Path(process.tpSelection*process.photonPrevalidationSequence*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
