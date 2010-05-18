
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
photonValidation.OutputFileName = 'PhotonValidationRelVal361_QCD_Pt_80_120.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 361 QCD_Pt_80_120
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V7-v1/0021/26A1878F-515D-DF11-BBDC-002618943966.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V7-v1/0020/D4EC346A-3E5D-DF11-9F77-00261894382D.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V7-v1/0020/B22217FE-455D-DF11-A958-0026189438FC.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V7-v1/0020/8E5D337C-445D-DF11-BA26-00261894390B.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V7-v1/0020/7C291ECE-3E5D-DF11-B466-0018F3D095EE.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V7-v1/0020/4881B652-3F5D-DF11-9F55-00261894388D.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V7-v1/0020/3E6C9F86-415D-DF11-AE55-001A92810AA8.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V7-v1/0020/0CD66A59-435D-DF11-91BB-001A92971B48.root'

        ),
    
    secondaryFileNames = cms.untracked.vstring(
        # official RelVal 361 QCD_Pt_80_120
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/F03CF367-3E5D-DF11-B7FC-002618943958.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/DCF4A25E-445D-DF11-8EF9-001A92971B68.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/DABDD206-415D-DF11-8390-001A92971B48.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/D69F3287-415D-DF11-B294-002618FDA28E.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/A836236D-3E5D-DF11-8F04-003048678F26.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/94F66BA6-425D-DF11-9798-0018F3D096AE.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/8622F5DD-455D-DF11-9BFF-001A92971B8E.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/82022975-445D-DF11-BC42-001A92811730.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/70404F50-3F5D-DF11-8041-002618943843.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/6C490614-425D-DF11-A915-001A92811728.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/50656520-455D-DF11-8754-001A92971BC8.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/4CA4F550-3F5D-DF11-B3B7-002618943874.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/36892FB3-3D5D-DF11-9D03-003048678F26.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/283539AB-435D-DF11-B094-003048678E94.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/24F0268F-515D-DF11-AAF5-001A92810AA8.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/222AE0C9-3E5D-DF11-A0D8-001A92971B5E.root',
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/16B8EC68-3E5D-DF11-A111-0026189438FC.root'

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


