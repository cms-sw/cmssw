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
process.GlobalTag.globaltag = 'START37_V5::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre1_PhotonJets_Pt_10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre1 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V5-v1/0001/C40068D9-226E-DF11-A4A0-0026189438DA.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V5-v1/0000/AC707922-DA6D-DF11-9DB9-0018F3D095FC.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V5-v1/0000/A08EC7FB-D76D-DF11-86CF-002618943976.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V5-v1/0000/0ABA7A67-D86D-DF11-9532-0026189438B8.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START37_V5-v1/0000/0A24E7FF-D76D-DF11-ABCD-00261894388A.root'

    ),


    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre1 RelValPhotonJets_Pt_10
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/FE31A3F6-DC6D-DF11-84C4-003048678A80.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/E6394C8E-D96D-DF11-91DD-00261894380B.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/D87C7231-EF6D-DF11-8D3C-002618943967.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/ACBCC154-D76D-DF11-87A5-0030486792F0.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/705DFCF5-D76D-DF11-82E3-003048679236.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/5C4DE9FF-D76D-DF11-984A-0026189438E7.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/4EA19DF0-D86D-DF11-9293-003048D25B68.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/3C0C7DBC-D46D-DF11-A62C-00304867C0EA.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/3AC5E403-D86D-DF11-9ADF-002618943937.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/189E1FEF-D76D-DF11-BFFF-002618943930.root'


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
