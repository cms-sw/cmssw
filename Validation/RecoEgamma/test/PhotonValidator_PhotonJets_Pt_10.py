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
process.GlobalTag.globaltag = 'START39_V3::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal3_10_0_pre2_PhotonJets_Pt_10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V3-v1/0061/0C1D5F27-F5E2-DF11-BDF9-001BFCDBD11E.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V3-v1/0058/E8E5D3DC-A8E2-DF11-BC36-0018F3D0967E.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V3-v1/0058/C82384CD-ADE2-DF11-A41A-003048679048.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V3-v1/0058/AAE1BDFD-A8E2-DF11-B4D8-001A928116F2.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V3-v1/0058/5439780B-A8E2-DF11-8CA4-0026189437E8.root'

    ),


    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0060/FE4AB0CF-EFE2-DF11-8265-0018F3D0967E.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/E48E2983-A8E2-DF11-A5B2-00261894396F.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/BA12D664-A7E2-DF11-A9E1-002618943874.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/B66684C0-A7E2-DF11-821B-002618943907.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/AA9D9583-A8E2-DF11-95A9-001A92971B08.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/88592F65-A7E2-DF11-8D90-0018F3D096E0.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/82E73C79-A8E2-DF11-9281-002618943900.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/8296126E-A7E2-DF11-B2A6-0026189438DD.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/78570266-AEE2-DF11-B050-003048678F74.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/681E7316-A9E2-DF11-83C1-002618FDA208.root',
        '/store/relval/CMSSW_3_10_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0058/2E5A1E47-A8E2-DF11-A6A7-0018F3D096CA.root'



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
