
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
process.GlobalTag.globaltag = 'START3X_V26::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal357_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 357 QCD_Pt_80_120
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V26-v1/0012/D07F9540-5F49-DF11-8AED-00304867926C.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V26-v1/0012/C69151EF-5F49-DF11-AAE8-002618943800.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V26-v1/0012/ACA5FA8B-6149-DF11-BB1F-002354EF3BD2.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V26-v1/0012/A0BEB3B0-6249-DF11-8D1B-001A928116F0.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V26-v1/0012/768E99BB-5D49-DF11-8061-002354EF3BE0.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V26-v1/0012/60ECE71E-5F49-DF11-9700-0018F3D096C2.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V26-v1/0012/5CDD4945-5D49-DF11-9355-002618943879.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V26-v1/0012/4C7F6C25-6949-DF11-830C-003048679266.root'
  
     ),
        ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 357 QCD_Pt_80_120
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0013/4403912A-6949-DF11-93E8-003048678F62.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/D8B72090-6249-DF11-BE75-00304867915A.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/C6AB1CB2-5D49-DF11-B02E-003048D3FC94.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/C4B5032C-5E49-DF11-98C5-00304867915A.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/B670F50E-5F49-DF11-8DB9-002618943829.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/B0F10C2F-6149-DF11-BE0C-001A928116F0.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/90555697-5C49-DF11-AFD7-00304866C398.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/8E5EB6FB-5E49-DF11-93D9-002618943896.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/8A5E01B6-5D49-DF11-BCD5-003048678B30.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/60370F52-5F49-DF11-9B48-003048D3FC94.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/5895E76C-6149-DF11-A2CA-00248C0BE016.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/462E1CC7-5F49-DF11-A4B1-00261894380D.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/40640429-5E49-DF11-811B-003048678FFA.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/402060CE-5F49-DF11-BE79-003048679296.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/2C2798F9-5E49-DF11-A2C1-0018F3D09644.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/206D2DD7-6149-DF11-94E3-0030486792B4.root',
        '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V26-v1/0012/06126627-5D49-DF11-BF0F-002618943843.root'

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


