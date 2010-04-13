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
process.GlobalTag.globaltag = 'MC_36Y_V4::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre6_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre6 single Photons pt=35GeV            

        '/store/relval/CMSSW_3_6_0_pre6/RelValSingleGammaPt35/GEN-SIM-RECO/MC_36Y_V4-v1/0011/4E0D95FD-AF44-DF11-8D35-00261894397E.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValSingleGammaPt35/GEN-SIM-RECO/MC_36Y_V4-v1/0011/02B15D76-4D45-DF11-A8DD-001A92971B38.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360pre6 single Photons pt=35GeV
        '/store/relval/CMSSW_3_6_0_pre6/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0011/E400FE8C-B044-DF11-8A6A-0026189438C0.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0011/D455FE91-B144-DF11-AB15-0026189438CF.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0011/BCF54BFA-AF44-DF11-AF4B-002618943913.root',
        '/store/relval/CMSSW_3_6_0_pre6/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V4-v1/0011/6C23155B-4D45-DF11-8439-001A92810AD6.root'
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
