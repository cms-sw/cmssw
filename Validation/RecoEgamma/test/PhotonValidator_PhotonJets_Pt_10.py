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
process.GlobalTag.globaltag = 'START39_V5::All'


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

photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre5_PhotonJets_Pt_10.root'

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
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V5-v1/0086/7400430B-70F5-DF11-9E21-001A92810AEA.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V5-v1/0085/FACBEC66-00F5-DF11-863D-00261894389A.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V5-v1/0085/B8E9135B-FAF4-DF11-88ED-001A92810ACA.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V5-v1/0085/9CDD17D7-06F5-DF11-98D8-0026189438D6.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V5-v1/0085/2AFB4590-00F5-DF11-A0BD-0018F3D096EA.root'
    ),


    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0088/2EEC7200-78F5-DF11-A5D3-00304867C0FC.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/FE33A9C1-FAF4-DF11-AC86-0018F3D096A6.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/F2A7396E-00F5-DF11-9D44-001A92971B74.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/EED7926B-FBF4-DF11-BA81-0018F3D095EA.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/B2F2D039-01F5-DF11-8D3D-002618943867.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/AA4E38DA-FFF4-DF11-8D8E-003048678B84.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/8E231794-F9F4-DF11-998C-002618FDA250.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/8C758B0C-F9F4-DF11-821C-00304867BEE4.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/76AD49EC-01F5-DF11-BD1B-002618FDA248.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/3C43EF10-FFF4-DF11-9149-002618943869.root',
        '/store/relval/CMSSW_3_10_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V5-v1/0085/260837F7-03F5-DF11-BE3C-002618943865.root'
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

process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
