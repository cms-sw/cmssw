
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
photonValidation.OutputFileName = 'PhotonValidationRelVal360_QCD_Pt_80_120.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360 QCD_Pt_80_120

        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0014/C8C0C5B1-FC49-DF11-A173-003048678BAC.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0014/BEC960FF-AB49-DF11-9EC9-003048678C26.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0014/4C98A821-BD49-DF11-866A-00261894385A.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0013/E46CD33B-9849-DF11-A862-001A928116F0.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0013/9E2E814D-AD49-DF11-9139-0018F3D09696.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0013/52CD0BD7-9649-DF11-A5C0-003048678FF8.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0013/2EFBB3F9-AD49-DF11-9A34-001A928116F2.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V4-v1/0013/0043CF79-9949-DF11-9541-002618943940.root'
        ),
    
    secondaryFileNames = cms.untracked.vstring(
        # official RelVal 360 QCD_Pt_80_120

        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0014/A6811FE2-AB49-DF11-B267-0018F3D096DA.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0014/967F08D0-B949-DF11-A421-00261894391F.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0014/5CEA2FD8-AE49-DF11-97B5-002618943963.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0014/3AB48B86-FC49-DF11-B805-003048678B76.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0014/2C793D20-BE49-DF11-B733-0026189438ED.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0014/00513E5C-AD49-DF11-989B-002354EF3BDA.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/F86EA908-9949-DF11-8D5C-0026189438ED.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/E4E5CBA9-9A49-DF11-8D5D-001A92971BC8.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/CEC94043-9649-DF11-9BC6-00304867BFAA.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/CCFEFEA5-9749-DF11-A2F3-0018F3D09616.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/5C419C8E-9949-DF11-927D-0018F3D09604.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/46142F5D-9949-DF11-A8A7-001A92971B1A.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/3EBA0794-9649-DF11-AA7B-0026189437EC.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/3A3FCFE2-9A49-DF11-82D4-002618943901.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/1C9068E0-9949-DF11-B2D4-0018F3D0965C.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/0CAAAD0C-9749-DF11-9A33-0018F3D09670.root',
        '/store/relval/CMSSW_3_6_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V4-v1/0013/061F0A51-9649-DF11-B59A-003048678D78.root'

    )
 )


photonPostprocessing.rBin = 48
## For gam Jet and higgs
photonValidation.eMax  = 500
photonValidation.etMax = 500
photonPostprocessing.eMax  = 500
photonPostprocessing.etMax = 500




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)


process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)


