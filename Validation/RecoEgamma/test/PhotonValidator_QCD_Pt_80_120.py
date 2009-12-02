
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
process.GlobalTag.globaltag = 'MC_31X_V9::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal335_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

# official RelVal 335 QCD_Pt_80_120

        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0008/E60CB8AD-12DC-DE11-B1F6-001731AF6933.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0007/E8707935-D2DB-DE11-8C44-003048678FE4.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0007/C803A4C0-D1DB-DE11-9D74-003048678FB2.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0007/B09AF79D-DBDB-DE11-913A-001A9281173E.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0007/7CA6455D-D3DB-DE11-8592-0018F3D09702.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0007/68F61534-D2DB-DE11-AD0A-002618943956.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0007/680FDDBD-D1DB-DE11-AB6C-0018F3D095EA.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V9-v1/0007/4A7CCDBE-D2DB-DE11-9F42-001A92971BBE.root'          

    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 335 QCD_Pt_80_120

        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0008/3C9BA2AC-12DC-DE11-B849-001731AF6789.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/F6CE4237-D2DB-DE11-BA20-001A92810ADE.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/EE41D155-D3DB-DE11-9356-00304867BEE4.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/E64B45AC-D9DB-DE11-9CB9-0026189437F9.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/BCE1E137-D2DB-DE11-87F8-0018F3D09702.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/96BFAC37-D2DB-DE11-9707-0018F3D09670.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/9216F9B6-D1DB-DE11-89BD-003048678FB2.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/82ED12BD-D1DB-DE11-ABD5-003048678CA2.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/7C4EF951-D3DB-DE11-ADCC-0026189438A2.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/469C5FBC-D1DB-DE11-9334-003048679168.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/4405B6C2-D1DB-DE11-BF98-0018F3D095EA.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/2E38A654-D3DB-DE11-AE2E-003048D3FC94.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/281EDA37-D2DB-DE11-9B3D-0018F3D09670.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/2476E5BE-D2DB-DE11-A06A-0018F3D09670.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/1AA8C65A-D3DB-DE11-BB2E-0018F3D09670.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/16D6FCBE-D2DB-DE11-B86B-0018F3D09670.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/1645A5BC-D1DB-DE11-85B7-0030486790FE.root',
        '/store/relval/CMSSW_3_3_5/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/0E4B80BB-D1DB-DE11-9B7B-003048678FE4.root'


     
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


