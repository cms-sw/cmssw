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
process.GlobalTag.globaltag = 'START37_V4::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre5_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre5 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V4-v1/0023/504F3BAD-8863-DF11-89ED-003048679076.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V4-v1/0022/BA657622-5363-DF11-B8CC-003048679164.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V4-v1/0022/26EDFAB8-5363-DF11-BAB3-002618943954.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V4-v1/0022/1C19D434-5563-DF11-AA43-00261894397E.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V4-v1/0022/10CCAFA3-5463-DF11-909C-0026189438FF.root'

    ),


    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre5 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/BC8024C0-5363-DF11-9D29-00304867918A.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/BAC58235-5663-DF11-ABF1-00261894390E.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/9AE78ABF-5363-DF11-93D4-00261894385A.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/8C893181-5263-DF11-B7EA-002618943951.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/826BAAA1-5463-DF11-AA19-002618943960.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/648B3641-5463-DF11-9A77-0026189438AA.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/4AA32B22-5363-DF11-BF73-003048678BE6.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/20078631-5563-DF11-BCEB-002618943946.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/10722BBF-5363-DF11-AF55-00304867BEDE.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/0289A7A2-5463-DF11-A1F9-00261894394B.root'

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
