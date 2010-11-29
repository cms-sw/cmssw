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
process.GlobalTag.globaltag = 'START39_V6::All'


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

photonValidation.OutputFileName = 'PhotonValidationRelVal394_PhotonJets_Pt_10.root'

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
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V6-v1/0003/DCC9A2F3-48F8-DF11-8F22-003048678C26.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V6-v1/0001/AE2D4E55-15F8-DF11-A21B-002618943875.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V6-v1/0001/04FFEA17-1BF8-DF11-990A-00304867920C.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V6-v1/0000/AEA3CA6E-FAF7-DF11-B14D-002354EF3BDB.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V6-v1/0000/6CCFF972-F6F7-DF11-9C3C-001A92810AE4.root'


    ),


    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0003/4475C1EE-48F8-DF11-AD72-001A92811726.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/EEB20FB9-16F8-DF11-BA95-003048678F02.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/C241E220-15F8-DF11-9A22-0018F3D09648.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/56F297A6-17F8-DF11-B043-0018F3D09704.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/34E63218-1BF8-DF11-9473-001A92810ACA.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/E24F84E0-FDF7-DF11-B1F0-001A928116FA.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/BE1FD7C7-F9F7-DF11-AC93-002354EF3BDB.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/A46B2475-F5F7-DF11-807C-001A92811718.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/94EF127A-F7F7-DF11-A4A3-002618943869.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/60FF3BE9-F1F7-DF11-8DD6-0030486792AC.root',
        '/store/relval/CMSSW_3_9_4/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/0CAE5B6B-FAF7-DF11-B9AE-001A92971BDA.root'



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
