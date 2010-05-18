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
process.GlobalTag.globaltag = 'MC_37Y_V3::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre4_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre4 single Photons pt=35GeV
        '/store/relval/CMSSW_3_7_0_pre4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V3-v1/0022/64D15484-A85D-DF11-88E6-0030486790A6.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V3-v1/0021/DA87DA85-775D-DF11-A7CA-0030486790B8.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V3-v1/0021/96CBE944-7C5D-DF11-899C-001A92811738.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre4 single Photons pt=35GeV
        '/store/relval/CMSSW_3_7_0_pre4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V3-v1/0022/78D04B7F-A85D-DF11-B09A-0026189438AC.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V3-v1/0021/6E3D373B-765D-DF11-BAEC-001A92811742.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V3-v1/0021/6889867E-7A5D-DF11-9437-0018F3D09616.root',
        '/store/relval/CMSSW_3_7_0_pre4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V3-v1/0021/2ACB0B85-775D-DF11-B8C8-002618943826.root'
    
    )
 )


photonPostprocessing.rBin = 48

## For single gamma pt = 35
photonValidation.eMax  = 300
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonValidation.dCotCutOn = False
photonValidation.dCotCutValue = 0.15

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)



process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
