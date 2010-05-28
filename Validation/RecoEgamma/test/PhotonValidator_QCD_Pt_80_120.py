
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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre5_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre5 QCD_Pt_80_120
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0024/B42538FB-8F63-DF11-B1DE-003048678B3C.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0022/F094F1B3-6363-DF11-B015-00261894391D.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0022/EA113D69-5F63-DF11-BAB0-0030486791BA.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0022/B4EBB6D1-6563-DF11-BEBC-00248C0BE018.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0022/AC86AD07-5B63-DF11-A4F1-003048678EE2.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0022/98981B99-6263-DF11-9A4C-00261894394B.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0022/6C14C433-5C63-DF11-B8DE-002618943950.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V4-v1/0022/0CC12EBC-5D63-DF11-AF72-003048678B14.root'
 
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre5 QCD_Pt_80_120
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0023/4035B165-8963-DF11-AC0D-00261894394D.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/F6F619EF-5B63-DF11-9916-002618943950.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/F4376802-5C63-DF11-863B-003048678EE2.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/EA398F49-6463-DF11-9BB3-003048678F74.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/E47C0F80-6163-DF11-97BA-003048678F74.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/D8C52ADF-5E63-DF11-921E-003048678BE6.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/B075873A-6563-DF11-AA34-0030486790A6.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/B051BD39-5E63-DF11-8790-00304867D838.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/A22AD296-6263-DF11-A501-00304867929E.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/A09EB794-6263-DF11-B7E2-002618943829.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/842F9753-6363-DF11-9C17-00261894390C.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/7A4A5045-5C63-DF11-902A-00248C55CC4D.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/4E18EEED-5F63-DF11-A83B-002618943923.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/2A32FFBB-5D63-DF11-85E7-003048678C9A.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/0CD2E708-5B63-DF11-8A11-002354EF3BDE.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/08ADF18F-6663-DF11-A94B-002618943861.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0022/06C4BA03-5B63-DF11-9ABF-003048D42D92.root'


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


