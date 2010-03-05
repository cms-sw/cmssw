
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
process.GlobalTag.globaltag = 'MC_3XY_V21::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal350_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 350 QCD_Pt_80_120
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0013/B27E46BF-3E13-DF11-A7EE-001A9281172C.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0013/44E856CB-3F13-DF11-8B01-001A92971B7C.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0013/3ACDFD75-4013-DF11-A204-001A92971BDC.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0013/3A52592E-3E13-DF11-8EAE-001A92810AEA.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0013/38DA44B8-3D13-DF11-B10D-0018F3D09642.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0013/14004A7D-6213-DF11-B250-001A92810AEE.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0012/EA2B34EC-3913-DF11-8E34-0026189438BC.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0012/DA4420C6-3813-DF11-84CC-003048679000.root'



        
    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 350 QCD_Pt_80_120

 
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/A2650844-3F13-DF11-9F8B-0018F3D096C6.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/7C862077-6213-DF11-A55E-0018F3D096DA.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/64717429-3E13-DF11-A79C-0018F3D096DC.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/5AB9E5C4-3E13-DF11-A5C6-0018F3D096AE.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/3A229928-3E13-DF11-A0F6-0018F3D09702.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/320738C4-3F13-DF11-B7EF-0018F3D09710.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/2882F7B5-4013-DF11-A8AB-001A92971B64.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/2437DA46-3F13-DF11-8BD2-001A92810ACA.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/20B66439-3D13-DF11-A811-001A92971B0C.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/16000C72-4013-DF11-B408-001A9281172E.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/0604FA28-3E13-DF11-81A6-001A92971B06.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/0453E0BD-3E13-DF11-AC08-001A92971B94.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0012/E4A7683D-3913-DF11-A8DA-002618943852.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0012/D6AF091A-3A13-DF11-AEFE-00304867901A.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0012/D0EDD332-3813-DF11-81D8-003048678F8E.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0012/BC3943E5-3813-DF11-AF6D-00304867901A.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0012/B8631430-3813-DF11-894F-00261894393C.root',
        '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0012/20C261E3-3913-DF11-AC49-002618FDA287.root'


        
     
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


