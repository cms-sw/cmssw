
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
process.GlobalTag.globaltag = 'START37_V0::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre1_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre1 QCD_Pt_80_120
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V0-v1/0015/C03F9395-E04C-DF11-BE29-00304867C04E.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V0-v1/0015/BA5B1196-E74C-DF11-BF0C-002618943983.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V0-v1/0015/9CD63685-DF4C-DF11-A9D3-0018F3D09698.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V0-v1/0015/92552DA9-E14C-DF11-B9C6-00248C0BE012.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V0-v1/0015/481B188A-E54C-DF11-A322-003048679048.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V0-v1/0015/2C556B6D-0E4D-DF11-84A5-0026189438A9.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V0-v1/0015/24A39B7C-DE4C-DF11-B3C2-003048679046.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V0-v1/0015/084758B1-E24C-DF11-8B43-00304867915A.root'
  
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre1 QCD_Pt_80_120
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/F47C2D94-E04C-DF11-A688-0018F3D09636.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/E2EF9CC1-E24C-DF11-8452-003048678B30.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/D019D6AD-E14C-DF11-A301-003048679070.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/CCF6BC0D-E04C-DF11-B838-002618943865.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/B858AC9D-E04C-DF11-BFAF-0018F3D0965E.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/B6EA4B7A-DE4C-DF11-AE86-003048678B86.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/B208A9C1-E84C-DF11-89E1-001BFCDBD100.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/B06EADEA-DE4C-DF11-9236-0026189437FE.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/9CC79D62-0E4D-DF11-8057-002618943833.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/747D158C-DF4C-DF11-98CC-001A9281173E.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/5A7DF5AD-E34C-DF11-A40A-00304867915A.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/2EF9F2AD-E14C-DF11-A4C6-003048679030.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/2E342A97-E04C-DF11-B62D-001A92971AAA.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/18777C94-E64C-DF11-B3E5-001A928116D2.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/14260313-E64C-DF11-853B-003048678FD6.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/0ADE5B28-E34C-DF11-ADF6-003048D3FC94.root',
        '/store/relval/CMSSW_3_7_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V0-v1/0015/0A18168E-DE4C-DF11-B2C7-0018F3D0960C.root'
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


