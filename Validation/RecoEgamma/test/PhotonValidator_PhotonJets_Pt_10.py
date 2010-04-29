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
process.GlobalTag.globaltag = 'START37_V1::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre2_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre2 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V1-v1/0018/00518909-F752-DF11-893B-00261894393C.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V1-v1/0017/EA1EDB09-8E52-DF11-8164-00261894395A.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V1-v1/0017/7474865A-8F52-DF11-B760-00304867900C.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V1-v1/0017/54ABDA5D-8F52-DF11-906F-00304867C0FC.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V1-v1/0017/1404656A-8F52-DF11-9BE6-00304867D836.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre2 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0018/CE944301-F752-DF11-8AEC-00261894395A.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/EC052548-9052-DF11-920B-002618943975.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/A8323159-8F52-DF11-A7BD-003048679046.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/A61C5C5F-8F52-DF11-BC1B-00304867D836.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/8E13BB58-8F52-DF11-9BAF-003048679006.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/6E5AE68A-8D52-DF11-BA03-003048678BB2.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/5EA44F00-8E52-DF11-9AF4-002618943875.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/4E89993E-9052-DF11-987D-002618943977.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/2416ADFC-8D52-DF11-863E-00248C55CC7F.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/22749E43-9052-DF11-9597-002618943908.root',
        '/store/relval/CMSSW_3_7_0_pre2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V1-v1/0017/14063685-8D52-DF11-BB7F-00261894398B.root'

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
