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
process.GlobalTag.globaltag = 'MC_38Y_V6::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre8_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre8 single Photons pt=35GeV
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V6-v1/0001/404F49D6-A78B-DF11-949B-00304867BF18.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V6-v1/0000/D293B3B0-558B-DF11-825C-00248C0BE014.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V6-v1/0000/9E294422-568B-DF11-80F4-00248C0BE014.root'  

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre8 single Photons pt=35GeV
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V6-v1/0001/A62F8B39-A38B-DF11-8E6C-00248C55CC9D.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V6-v1/0000/D0100171-548B-DF11-BBDB-003048678B12.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V6-v1/0000/4CEAC811-558B-DF11-A502-001A928116B2.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V6-v1/0000/2AA55EF2-548B-DF11-8A2C-001A92810AD0.root'
        
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
