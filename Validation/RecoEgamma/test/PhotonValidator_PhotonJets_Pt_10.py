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
process.GlobalTag.globaltag = 'START39_V2::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre7_PhotonJets_Pt_10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V2-v1/0045/FA57FD90-B5D3-DF11-AB6A-00261894395A.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V2-v1/0044/5882CD54-58D3-DF11-9022-003048678C9A.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V2-v1/0043/BC43DE48-51D3-DF11-BE95-0026189438D2.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V2-v1/0043/52ABDBC1-50D3-DF11-8158-00261894392C.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V2-v1/0043/20EDD0B5-4DD3-DF11-8F67-00261894397D.root'


    ),


    secondaryFileNames = cms.untracked.vstring(
 '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0045/18818A13-B4D3-DF11-84DE-00261894393B.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/FCE668CE-58D3-DF11-8045-00261894387E.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0043/D0AEF746-51D3-DF11-B5E9-002618943856.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0043/B60F6627-4DD3-DF11-AE24-00261894386B.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0043/8C511CAF-4CD3-DF11-8DE0-0030486792B8.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0043/42136049-51D3-DF11-AC52-003048679030.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0043/403CF3C5-56D3-DF11-BC3F-003048678F0C.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0043/3A684FBD-50D3-DF11-AE6E-002618943866.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0043/34B31EB6-4DD3-DF11-BFFD-002354EF3BE6.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0043/30FAEA3F-50D3-DF11-9AFD-002618943956.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0043/247EA0C1-50D3-DF11-9266-003048678BE6.root'


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
