
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
process.GlobalTag.globaltag = 'START3X_V25::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal355_QCD_Pt_80_120.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 355 QCD_Pt_80_120
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V25-v1/0007/B89A2C9C-1138-DF11-BD56-002618943894.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V25-v1/0006/FC91559F-D737-DF11-92A4-0018F3D0969C.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V25-v1/0006/FAAD86BD-D237-DF11-BB05-002618943979.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V25-v1/0006/A4F64512-D137-DF11-B435-003048678B76.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V25-v1/0006/7E0C689B-D137-DF11-85BC-00261894393D.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V25-v1/0006/64EAAFA4-D037-DF11-8520-003048678CA2.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V25-v1/0006/2E1BF5CF-D337-DF11-A497-0026189438AE.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V25-v1/0006/10469EA9-D037-DF11-8BF7-003048678FE0.root'


     ),

    secondaryFileNames = cms.untracked.vstring(
# official RelVal 355 QCD_Pt_80_120

        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/FCBC48A0-D637-DF11-B42D-001A92810AA4.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/F69C5BA5-D037-DF11-8D1A-0018F3D09708.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/E4BC42A1-D037-DF11-85ED-002618943863.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/BED1B391-1138-DF11-AF7F-002618FDA265.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/847A2E16-D137-DF11-94A9-001A92811720.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/6EA53D10-D137-DF11-96AD-002618943905.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/62D6D8A5-D037-DF11-B5A0-001A928116DA.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/58359B97-D137-DF11-9659-0026189437F5.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/56AB8636-D337-DF11-A37D-002618943826.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/5463AAA0-D037-DF11-BF69-0018F3D0967A.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/4063D99F-D437-DF11-9FD2-0026189438AE.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/3E5A4D87-CF37-DF11-B831-0018F3D09660.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/343F1FA5-D237-DF11-98CD-0026189438DB.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/340187A2-D337-DF11-8FA4-001A9281172E.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/1EAA0D95-D137-DF11-8647-0026189438BC.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/1E8D8995-D137-DF11-9080-0026189438D2.root',
        '/store/relval/CMSSW_3_5_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V25-v1/0006/1695AC17-D237-DF11-8D51-00261894390A.root'

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


