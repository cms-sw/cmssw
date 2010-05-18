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
process.GlobalTag.globaltag = 'START36_V7::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal361_PhotonJets_Pt_10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 361 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V7-v1/0020/B846DFCB-355D-DF11-B8F0-002618FDA237.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V7-v1/0020/98D5403F-335D-DF11-8ED2-0018F3D095FC.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V7-v1/0020/4C8AE4CC-345D-DF11-B165-001A92971BDC.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V7-v1/0020/4A6DD46A-525D-DF11-B41B-0018F3D096A4.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START36_V7-v1/0020/1E64894A-345D-DF11-90D8-00261894393C.root'
    ),


    secondaryFileNames = cms.untracked.vstring(
# official RelVal 361 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0021/C29A6066-525D-DF11-A176-001A92811714.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/F4C61D4B-345D-DF11-9A0F-001BFCDBD190.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/E4FBABB1-335D-DF11-A626-00304867920C.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/C4B51D4B-345D-DF11-B98C-002618943865.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/B20971D1-365D-DF11-B8C9-0018F3C3E3A6.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/A8F78EC7-345D-DF11-91FE-00261894397E.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/9C88733C-335D-DF11-B0DD-0018F3D09628.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/8C98234E-355D-DF11-A288-002618943849.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/5ABA071A-315D-DF11-B127-0018F3D09688.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/3A43A14B-345D-DF11-90C2-002618FDA28E.root',
        '/store/relval/CMSSW_3_6_1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/0E8C4A52-355D-DF11-8E77-0018F3D0968C.root'

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
